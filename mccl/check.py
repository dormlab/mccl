"""Environment verification for mccl.

Run `mccl check` to confirm Thunderbolt and (optionally) Tailscale are
configured correctly before launching a multi-node job.

The checks are all read-only — nothing is modified.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple

CheckResult = Tuple[bool, str]


def _run(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def _local_ipv4_by_iface():
    """Return {iface: [ipv4, ...]} from `ifconfig` output."""
    rc, txt = _run(["ifconfig"])
    if rc != 0:
        return {}
    res: dict[str, list[str]] = {}
    cur = None
    for line in txt.splitlines():
        if not line.startswith("\t") and ":" in line and " " not in line.split(":")[0]:
            cur = line.split(":")[0]
            res.setdefault(cur, [])
        elif cur and line.lstrip().startswith("inet "):
            ip = line.split()[1]
            if ":" not in ip:
                res[cur].append(ip)
    return res


def check_platform() -> CheckResult:
    if platform.system() != "Darwin":
        return False, f"OS is {platform.system()}, need Darwin (macOS)"
    if platform.machine() not in ("arm64", "aarch64"):
        return False, f"arch is {platform.machine()}, need arm64 (Apple Silicon)"
    return True, f"macOS {platform.mac_ver()[0]} on {platform.machine()}"


def check_python() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 11)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else " (need >= 3.11)")


def check_xcode_cli() -> CheckResult:
    rc, out = _run(["xcrun", "--show-sdk-path"])
    if rc != 0:
        return False, "xcrun missing — run `xcode-select --install`"
    return True, f"SDK at {out.strip()}"


def check_mccl_loaded() -> CheckResult:
    try:
        import mccl  # noqa: F401
        import torch.distributed as dist
        if "mccl" not in dist.Backend.backend_type_map:
            return False, "mccl imports but backend not registered"
        return True, f"mccl {mccl.__version__}, backend 'mccl' registered"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_thunderbolt() -> CheckResult:
    """At least one Thunderbolt interface with an assigned IPv4."""
    rc, out = _run(["networksetup", "-listallhardwareports"])
    if rc != 0:
        return False, "networksetup unavailable"
    tb_ifaces = []
    block = ""
    for line in out.splitlines() + [""]:
        if line.startswith("Hardware Port"):
            block = line + "\n"
        elif line.startswith("Device:") and block:
            block += line + "\n"
        elif not line.strip() and "Thunderbolt" in block and "Device:" in block:
            dev = block.split("Device:")[1].split()[0].strip()
            tb_ifaces.append(dev)
            block = ""
        else:
            block += line + "\n"
    ips = _local_ipv4_by_iface()
    with_ip = {i: ips[i] for i in tb_ifaces if ips.get(i)}
    if not tb_ifaces:
        return False, "no Thunderbolt interfaces detected"
    if not with_ip:
        return False, (f"TB interfaces {tb_ifaces} have no IPv4 assigned. "
                       f"Configure System Settings → Network → Thunderbolt Bridge "
                       f"(or each TB port) with a static address in a /24 like "
                       f"192.168.10x.y")
    return True, ", ".join(f"{i}={','.join(with_ip[i])}" for i in with_ip)


def check_tailscale() -> CheckResult:
    """Tailscale presence is advisory — used for control plane / SSH, not data."""
    if not shutil.which("tailscale"):
        return False, "tailscale not installed (advisory — needed for control-plane / SSH)"
    rc, out = _run(["tailscale", "status"])
    if rc != 0:
        return False, "tailscale present but `tailscale status` failed (logged out?)"
    peers = [l for l in out.splitlines() if l.strip() and not l.startswith("#")]
    return True, f"tailscale up, {len(peers)} known peers"


def check_peers(peer_env: str | None = None) -> CheckResult:
    """Optional: ping each peer IP from MCCL_PEERS env to confirm reachability."""
    peer_env = peer_env or os.environ.get("MCCL_PEERS", "")
    if not peer_env:
        return True, "skipped (set MCCL_PEERS=ip1,ip2,... to test)"
    targets = [p.strip() for p in peer_env.split(",") if p.strip()]
    failed = []
    for t in targets:
        rc, _ = _run(["ping", "-c", "1", "-t", "2", t], timeout=4.0)
        if rc != 0:
            failed.append(t)
    if failed:
        return False, f"{len(failed)}/{len(targets)} unreachable: {failed}"
    return True, f"all {len(targets)} peers reachable"


CHECKS = [
    ("platform",        check_platform),
    ("python",          check_python),
    ("xcode-cli",       check_xcode_cli),
    ("mccl-backend",    check_mccl_loaded),
    ("thunderbolt",     check_thunderbolt),
    ("tailscale",       check_tailscale),
    ("peer-reach",      check_peers),
]


def run_all() -> int:
    print("mccl setup check\n")
    required = {"platform", "python", "xcode-cli", "thunderbolt"}
    failures = 0
    for name, fn in CHECKS:
        ok, msg = fn()
        is_required = name in required
        mark = "OK" if ok else ("FAIL" if is_required else "warn")
        bullet = " " if ok else "!"
        print(f"  [{mark:>4}] {bullet} {name:<14} {msg}")
        if not ok and is_required:
            failures += 1
    print()
    if failures:
        print(f"{failures} required check(s) failed.")
        return 1
    print("required checks pass — backend should run.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
