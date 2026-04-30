"""
Unified configuration for MCCL — Distributed Metal GPU Runtime.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict


@dataclasses.dataclass
class DistroConfig:
    """All distro tunables in one place."""

    # ── Cluster ─────────────────────────────────────────────────────────
    node_id: int = 0
    num_nodes: int = 3
    control_port: int = 29600

    # ── DMEM ────────────────────────────────────────────────────────────
    chunk_bytes: int = 1 * 1024 * 1024     # Max RDMA chunk size (1 MB)
    cq_depth: int = 256                    # Completion queue depth
    connect_timeout_ms: int = 30000

    # ── Compute ─────────────────────────────────────────────────────────
    fast_math: bool = True
    shader_path: str = ""

    # ── Runtime ─────────────────────────────────────────────────────────
    max_queue_depth: int = 1024
    log_level: str = "WARN"

    # ── Env var name → field mapping ────────────────────────────────────
    _ENV_MAP: Dict[str, str] = dataclasses.field(
        default_factory=dict, repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._ENV_MAP = {
            "DISTRO_NODE_ID": "node_id",
            "DISTRO_NUM_NODES": "num_nodes",
            "DISTRO_CONTROL_PORT": "control_port",
            "DISTRO_CHUNK_BYTES": "chunk_bytes",
            "DISTRO_CQ_DEPTH": "cq_depth",
            "DISTRO_CONNECT_TIMEOUT_MS": "connect_timeout_ms",
            "DISTRO_FAST_MATH": "fast_math",
            "DISTRO_SHADER_PATH": "shader_path",
            "DISTRO_MAX_QUEUE_DEPTH": "max_queue_depth",
            "DISTRO_LOG_LEVEL": "log_level",
        }
        self._validate()

    def _validate(self) -> None:
        if self.node_id < 0:
            raise ValueError(f"node_id must be >= 0, got {self.node_id}")
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        if self.chunk_bytes < 1:
            raise ValueError(f"chunk_bytes must be >= 1, got {self.chunk_bytes}")

    @classmethod
    def from_env(cls) -> "DistroConfig":
        cfg = cls()
        for env_key, field_name in cfg._ENV_MAP.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            fld = _field_type(cfg, field_name)
            setattr(cfg, field_name, _coerce(raw, fld))
        return cfg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistroConfig":
        known = {f.name for f in dataclasses.fields(cls) if f.name != "_ENV_MAP"}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_env(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for env_key, field_name in self._ENV_MAP.items():
            val = getattr(self, field_name)
            s = "1" if isinstance(val, bool) and val else (
                "0" if isinstance(val, bool) else str(val))
            if s and s != "":
                os.environ[env_key] = s
                out[env_key] = s
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "_ENV_MAP"
        }


def _field_type(obj: Any, name: str) -> type:
    for f in dataclasses.fields(obj):
        if f.name == name:
            return f.type
    return str


def _coerce(raw: str, target_type: Any) -> Any:
    if target_type is bool or target_type == "bool":
        return raw not in ("0", "false", "FALSE", "no", "NO", "")
    if target_type is int or target_type == "int":
        return int(raw)
    if target_type is float or target_type == "float":
        return float(raw)
    return raw
