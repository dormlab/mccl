#!/usr/bin/env python3
"""
distro CLI — submit jobs, query cluster status, manage nodes.

Usage::

    # Show cluster status
    distro status

    # Submit a training job
    distro submit --name "llama-7b-finetune" --nodes 3 --memory 16GB

    # List active jobs
    distro jobs

    # Cancel a job
    distro cancel --job-id 42

    # Watch the scheduler queue live
    distro watch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


# ── These resolve once distro._C is built ──────────────────────────────────
try:
    import distro
    from distro._C import init_dmem, shutdown_dmem, get_stats
    HAS_DISTRO = True
except ImportError:
    HAS_DISTRO = False


def cmd_status(args) -> None:
    """Print cluster node status."""
    if not HAS_DISTRO:
        print("distro native extension not built. Run: pip install -e .")
        sys.exit(1)

    stats = get_stats()
    print("┌─ distro Cluster Status ─────────────────────────────────┐")
    print(f"│  DMEM puts:  {stats['total_put_bytes'] / 1e9:>8.2f} GB  "
          f"({stats['total_put_ops']} ops)")
    print(f"│  DMEM gets:  {stats['total_get_bytes'] / 1e9:>8.2f} GB  "
          f"({stats['total_get_ops']} ops)")
    print(f"│  Errors:     {stats['total_errors']}")
    print("└─────────────────────────────────────────────────────────┘")


def cmd_submit(args) -> None:
    """Submit a job to the cluster scheduler."""
    if not HAS_DISTRO:
        print("distro native extension not built. Run: pip install -e .")
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
    print(f"\n  distro submit --name {args.name} --nodes {args.nodes} "
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


def main():
    parser = argparse.ArgumentParser(
        description="distro — Distributed Metal GPU Runtime CLI",
        prog="distro",
    )
    sub = parser.add_subparsers(dest="command")

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
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
