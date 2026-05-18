"""Register the MCCL c10d backend with torch.distributed.

Importing this module is sufficient to make `init_process_group(backend="mccl")`
work. It is idempotent — repeated imports do nothing after the first.
"""
from __future__ import annotations

import datetime as _dt
import os as _os
import warnings as _warnings

_REGISTERED = False

DEFAULT_IFACE_PRIORITY = (
    "192.168.101.",
    "192.168.102.",
    "192.168.103.",
)


def set_iface_priority(prefixes):
    """Set the IPv4 prefix priority used to pick the best peer route.

    Earlier prefixes win. Pass before init_process_group. Example:
        mccl.set_iface_priority(["192.168.103.", "192.168.102.", "192.168."])
    """
    _os.environ["MCCL_IFACE_PRIORITY"] = ",".join(prefixes)


def register() -> bool:
    """Register the 'mccl' backend. Returns True if registration succeeded."""
    global _REGISTERED
    if _REGISTERED:
        return True

    _os.environ.setdefault("MCCL_IFACE_PRIORITY", ",".join(DEFAULT_IFACE_PRIORITY))

    try:
        import torch.distributed as dist
    except ImportError:
        _warnings.warn("torch.distributed not available; skipping mccl registration",
                       RuntimeWarning, stacklevel=2)
        return False

    try:
        from mccl._C import _create_process_group_mccl
    except ImportError as e:
        _warnings.warn(
            f"mccl._C native extension missing _create_process_group_mccl: {e}",
            RuntimeWarning, stacklevel=2)
        return False

    def _create(prefix_store, rank, world_size, timeout):
        td = timeout if isinstance(timeout, _dt.timedelta) else _dt.timedelta(seconds=1800)
        return _create_process_group_mccl(prefix_store, rank, world_size, td)

    # MCCL is MPS-only — the PG body runs Metal kernels and rejects CPU tensors
    # via check_mps_ (see ProcessGroupMCCL.cpp). "cpu" stays in the device list
    # purely as a dispatcher-routing artifact: MPSFallback.cpp redispatches the
    # c10d ops from the MPS key into the CPU key, where torch ships a generic
    # shim that pulls the ProcessGroup off the op stack and calls our PG. No
    # CPU compute runs through MCCL.
    try:
        dist.Backend.register_backend("mccl", _create, devices=["cpu", "mps"])
    except TypeError:
        dist.Backend.register_backend("mccl", _create)

    _REGISTERED = True
    return True
