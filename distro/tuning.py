"""
Optional performance presets for MCCL (set ``os.environ`` before ``init_process_group``).
"""
from __future__ import annotations

import os


def apply_thunderbolt_production_defaults(
    *,
    training_defaults: bool = False,
) -> None:
    """Opt in to Thunderbolt-oriented TCP defaults (large buffers, 16MB chunks).

    Sets ``DISTRO_LINK_PROFILE=thunderbolt`` if unset. The native layer then applies:

    - Larger default socket buffers (see ``Connection::configure_socket``).
    - At least **16 MiB** ``DISTRO_CHUNK_BYTES`` if that env var is unset.

    If ``training_defaults`` is True, also sets (when unset) ``DISTRO_OVERLAP_COMM=1``
    and ``DDP_BUCKET_MB=512`` for PyTorch DDP workloads.

    Call **before** ``dist.init_process_group(backend="distro", ...)`` on every node.
    """
    os.environ.setdefault("DISTRO_LINK_PROFILE", "thunderbolt")
    if training_defaults:
        os.environ.setdefault("DISTRO_OVERLAP_COMM", "1")
        os.environ.setdefault("DDP_BUCKET_MB", "512")


__all__ = ["apply_thunderbolt_production_defaults"]
