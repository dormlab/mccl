"""distro — MCCL c10d backend for Apple Silicon."""
from __future__ import annotations

import platform
import warnings

from distro.version import __version__, COMPATIBILITY_MATRIX
from distro.config import DistroConfig


def _check_platform() -> bool:
    if platform.system() != "Darwin":
        warnings.warn("distro is designed for macOS on Apple Silicon.",
                      RuntimeWarning, stacklevel=2)
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        warnings.warn("distro requires Apple Silicon (arm64).",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


_platform_ok = _check_platform()

if _platform_ok:
    try:
        import distro._C  # noqa: F401
        from distro._backend import register as _register
        _register()
    except ImportError as e:
        warnings.warn(f"distro native extension not loaded: {e}",
                      RuntimeWarning, stacklevel=2)


__all__ = ["__version__", "COMPATIBILITY_MATRIX", "DistroConfig"]
