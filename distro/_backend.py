"""Register the MCCL c10d backend with torch.distributed.

Importing this module is sufficient to make `init_process_group(backend="mccl")`
work. It is idempotent — repeated imports do nothing after the first.
"""
from __future__ import annotations

import datetime as _dt
import warnings as _warnings

_REGISTERED = False


def register() -> bool:
    """Register the 'mccl' backend. Returns True if registration succeeded."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch.distributed as dist
    except ImportError:
        _warnings.warn("torch.distributed not available; skipping mccl registration",
                       RuntimeWarning, stacklevel=2)
        return False

    try:
        from distro._C import _create_process_group_mccl
    except ImportError as e:
        _warnings.warn(
            f"distro._C native extension missing _create_process_group_mccl: {e}",
            RuntimeWarning, stacklevel=2)
        return False

    def _create(prefix_store, rank, world_size, timeout):
        td = timeout if isinstance(timeout, _dt.timedelta) else _dt.timedelta(seconds=1800)
        return _create_process_group_mccl(prefix_store, rank, world_size, td)

    # Newer torch (≥2.1) accepts a `devices` kwarg; fall back if not.
    try:
        dist.Backend.register_backend("mccl", _create, devices=["cpu", "mps"])
    except TypeError:
        dist.Backend.register_backend("mccl", _create)

    _REGISTERED = True
    return True
