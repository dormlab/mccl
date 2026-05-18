"""mccl — MCCL c10d backend for Apple Silicon."""
from __future__ import annotations

import os
import platform
import sys
import warnings

# Disables torch's MPS commitAndContinue (see _backend.py for why).
# Must be set before torch.mps initializes.
os.environ.setdefault("PYTORCH_MPS_TRACE_SIGNPOSTS", "1")
if "torch" in sys.modules:
    warnings.warn(
        "mccl was imported after torch — set PYTORCH_MPS_TRACE_SIGNPOSTS=1 "
        "in your launcher to avoid DDP races.",
        RuntimeWarning, stacklevel=2)

from mccl.version import __version__, COMPATIBILITY_MATRIX
from mccl.config import DistroConfig


def _check_platform() -> bool:
    if platform.system() != "Darwin":
        warnings.warn("mccl is designed for macOS on Apple Silicon.",
                      RuntimeWarning, stacklevel=2)
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        warnings.warn("mccl requires Apple Silicon (arm64).",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


_platform_ok = _check_platform()

from mccl._backend import set_iface_priority  # noqa: E402

if _platform_ok:
    try:
        import mccl._C  # noqa: F401
        from mccl._backend import register as _register
        _register()
    except ImportError as e:
        warnings.warn(f"mccl native extension not loaded: {e}",
                      RuntimeWarning, stacklevel=2)


__all__ = ["__version__", "COMPATIBILITY_MATRIX", "DistroConfig", "set_iface_priority"]
