"""
distro — Distributed Metal GPU Runtime for Apple Silicon clusters.

Usage::

    import distro

    # Initialize the distributed memory manager
    distro.init_dmem(node_id=0, num_peers=3)

    # Register a Metal buffer for RDMA access
    region_id = distro.register_region(addr=..., length=..., flags=0x03)

    # RDMA write to a peer
    wr_id = distro.put(target_node=1, target_region=5, offset=0,
                       src_addr=..., length=1024)
    distro.poll_completion(wr_id)

    # Shutdown
    distro.shutdown_dmem()
"""
from __future__ import annotations

from distro.version import __version__, COMPATIBILITY_MATRIX
from distro.config import DistroConfig
from distro.tuning import apply_thunderbolt_production_defaults

import platform
import warnings
from typing import Any, Dict, Optional

# ── Platform check ───────────────────────────────────────────────────────

def _check_platform():
    if platform.system() != "Darwin":
        warnings.warn(
            "distro is designed for macOS on Apple Silicon.",
            RuntimeWarning, stacklevel=2,
        )
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        warnings.warn(
            "distro requires Apple Silicon (arm64).",
            RuntimeWarning, stacklevel=2,
        )
        return False
    return True

_platform_ok = _check_platform()
_C = None

if _platform_ok:
    try:
        from distro._C import (
            init_dmem,
            shutdown_dmem,
            register_region,
            unregister_region,
            put,
            get,
            poll_completion,
            drain_pending,
            get_stats,
            metal_kernels_init,
            metal_sync,
        )
        _C = True
        try:
            from distro._backend import register as _register_mccl_backend
            _register_mccl_backend()
        except Exception as _e:
            warnings.warn(
                f"failed to register mccl c10d backend: {_e}",
                RuntimeWarning, stacklevel=2,
            )
    except ImportError as e:
        warnings.warn(
            f"distro native extension not found: {e}",
            RuntimeWarning, stacklevel=2,
        )

# ── Public API ───────────────────────────────────────────────────────────

__all__ = [
    "__version__",
    "COMPATIBILITY_MATRIX",
    "DistroConfig",
    "init_dmem",
    "shutdown_dmem",
    "register_region",
    "unregister_region",
    "put",
    "get",
    "poll_completion",
    "drain_pending",
    "get_stats",
    "metal_kernels_init",
    "metal_sync",
    "apply_thunderbolt_production_defaults",
]
