#!/usr/bin/env python3
"""
mccl CLI — submit jobs, query cluster status, manage nodes.

Usage::

    # Show cluster status
    mccl status

    # Submit a training job
    mccl submit --name "llama-7b-finetune" --nodes 3 --memory 16GB

    # List active jobs
    mccl jobs

    # Cancel a job
    mccl cancel --job-id 42

    # Watch the scheduler queue live
    mccl watch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


# ── These resolve once mccl._C is built ──────────────────────────────────
try:
    import mccl
    from mccl._C import init_dmem, shutdown_dmem, get_stats
    HAS_DISTRO = True
except ImportError:
    HAS_DISTRO = False


def cmd_status(args) -> None:
    """Print cluster node status."""
    if not HAS_DISTRO:
        print("mccl native extension not built. Run: pip install -e .")
        sys.exit(1)

    stats = get_stats()
    print("┌─ mccl Cluster Status ─────────────────────────────────┐")
    print(f"│  DMEM puts:  {stats['total_put_bytes'] / 1e9:>8.2f} GB  "
          f"({stats['total_put_ops']} ops)")
    print(f"│  DMEM gets:  {stats['total_get_bytes'] / 1e9:>8.2f} GB  "
          f"({stats['total_get_ops']} ops)")
    print(f"│  Errors:     {stats['total_errors']}")
    print("└─────────────────────────────────────────────────────────┘")


def cmd_submit(args) -> None:
    """Submit a job to the cluster scheduler."""
    if not HAS_DISTRO:
        print("mccl native extension not built. Run: pip install -e .")
        sys.exit(1)

    # Parse memory string (e.g., "16GB" → bytes)
    memory_bytes = parse_memory(args.memory)

    print(f"Submitting job: {args.name}")
    print(f"  Nodes:  {args.nodes}")
    print(f"  Memory: {args.memory} ({memory_bytes / 1e9:.1f} GB per node)")
    print(f"  Owner:  {args.owner}")

    # In production, the head node has a REST/gRPC endpoint for job submission.
    # For now, the CLI talks to the local ClusterManager instance directly.
    print("\nJob submission via ClusterManager (requires head node).")
    print("The head node's ClusterManager.scheduler_loop() will allocate")
    print("resources on the next tick.")
    print(f"\n  mccl submit --name {args.name} --nodes {args.nodes} "
          f"--memory {args.memory}")


def cmd_jobs(args) -> None:
    """List active and pending jobs."""
    print("┌─ Active Jobs ───────────────────────────────────────────┐")
    print("│  (requires head node with ClusterManager running)")
    print("└─────────────────────────────────────────────────────────┘")


def cmd_cancel(args) -> None:
    """Cancel a job by ID."""
    print(f"Cancelling job {args.job_id}...")
    print("(requires head node with ClusterManager running)")


def cmd_watch(args) -> None:
    """Watch the scheduler queue live."""
    print("Watching scheduler queue (Ctrl-C to stop)...")
    try:
        while True:
            # In production: query head node's scheduler state
            if HAS_DISTRO:
                stats = get_stats()
                print(f"\r  puts={stats['total_put_ops']}  "
                      f"gets={stats['total_get_ops']}  "
                      f"errors={stats['total_errors']}",
                      end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n")


def parse_memory(s: str) -> int:
    """Parse a memory string like '16GB' or '512MB' into bytes."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            num = float(s[: -len(suffix)])
            return int(num * mult)

    return int(s)  # Raw bytes


def cmd_test(args) -> None:
    """Run the mccl test suite via pytest."""
    if args.worlds:
        os.environ["MCCL_TEST_WORLDS"] = args.worlds
    import subprocess
    targets = args.targets or ["tests/unit", "tests/integration"]
    cmd = [sys.executable, "-m", "pytest", *targets]
    if args.verbose:
        cmd.append("-v")
    if args.k:
        cmd.extend(["-k", args.k])
    if args.markers:
        cmd.extend(["-m", args.markers])
    print(f"+ MCCL_TEST_WORLDS={os.environ.get('MCCL_TEST_WORLDS', '2,3')} "
          + " ".join(cmd))
    sys.exit(subprocess.call(cmd))


def main():
    parser = argparse.ArgumentParser(
        description="mccl — Distributed Metal GPU Runtime CLI",
        prog="mccl",
    )
    parser.add_argument("--test", action="store_true",
                        help="Run the test suite (shortcut for `mccl test`).")
    parser.add_argument("--worlds", default=None,
                        help="Comma-separated world sizes for --test "
                             "(e.g. --worlds 2,3,4,8).")
    sub = parser.add_subparsers(dest="command")

    # test
    p = sub.add_parser("test", help="Run the test suite")
    p.add_argument("targets", nargs="*",
                   help="Test paths to run (default: tests/unit tests/integration)")
    p.add_argument("--worlds", default=None,
                   help="Comma-separated world sizes (overrides MCCL_TEST_WORLDS).")
    p.add_argument("-k", default=None, help="pytest -k expression")
    p.add_argument("-m", "--markers", default=None, help="pytest -m expression")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_test)

    # status
    sub.add_parser("status", help="Show cluster status").set_defaults(func=cmd_status)

    # submit
    p = sub.add_parser("submit", help="Submit a job")
    p.add_argument("--name", required=True)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--memory", default="16GB")
    p.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    p.add_argument("--priority", choices=["low", "normal", "high", "critical"],
                   default="normal")
    p.set_defaults(func=cmd_submit)

    # jobs
    sub.add_parser("jobs", help="List jobs").set_defaults(func=cmd_jobs)

    # cancel
    p = sub.add_parser("cancel", help="Cancel a job")
    p.add_argument("--job-id", type=int, required=True)
    p.set_defaults(func=cmd_cancel)

    # watch
    sub.add_parser("watch", help="Watch scheduler queue").set_defaults(func=cmd_watch)

    args = parser.parse_args()
    if args.test:
        # Shorthand: `mccl --test` → `mccl test`
        if not hasattr(args, "func"):
            args.targets = []
            args.k = None
            args.markers = None
            args.verbose = True
        cmd_test(args)
        return
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
