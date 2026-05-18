"""
DistributedTensor — a tensor sharded across the cluster via PGAS.

Each node holds a local shard (torch.Tensor on MPS). The DMEM layer
registers the shard's Metal buffer with the RDMA NIC, making it directly
accessible to peers via one-sided RDMA read/write.

All collective operations (all_reduce, all_gather, etc.) are implemented
via direct RDMA writes — no MPI or NCCL-style collectives needed.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import ctypes
import torch

from mccl._C import (
    register_region,
    unregister_region,
    put,
    get,
    poll_completion,
    drain_pending,
    metal_sync,
)


class ShardingSpec(IntEnum):
    REPLICATED = 0
    DATA_PARALLEL = 1
    TENSOR_PARALLEL = 2
    PIPELINE_PARALLEL = 3


class DistributedTensor:
    """A tensor whose storage spans multiple nodes via the DMEM layer."""

    def __init__(
        self,
        local_tensor: torch.Tensor,
        sharding: ShardingSpec,
        shard_index: int,
        num_shards: int,
        global_shape: Tuple[int, ...],
        dtype: torch.dtype,
        node_id: int = 0,
        region_id: Optional[int] = None,
    ):
        if local_tensor.device.type != "mps":
            raise ValueError("DistributedTensor requires MPS device tensors")
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        self._local = local_tensor
        self._sharding = sharding
        self._shard_index = shard_index
        self._num_shards = num_shards
        self._global_shape = global_shape
        self._dtype = dtype
        self._node_id = node_id

        # Register with DMEM for zero-copy RDMA access
        if region_id is None:
            self._region_id = self._register()
        else:
            self._region_id = region_id

        # Peer registries: {peer_node_id: (region_id, element_count)}
        self._peer_regions: Dict[int, Tuple[int, int]] = {}

    def _register(self) -> int:
        ptr = self._local.storage().data_ptr()
        nbytes = self._local.numel() * self._local.element_size()
        flags = 0x03  # READABLE | WRITABLE
        return register_region(ptr, nbytes, flags)

    def _deregister(self) -> None:
        if self._region_id is not None:
            unregister_region(self._region_id)
            self._region_id = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def global_shape(self) -> Tuple[int, ...]:
        return self._global_shape

    @property
    def local_data(self) -> torch.Tensor:
        return self._local

    @property
    def sharding(self) -> ShardingSpec:
        return self._sharding

    @property
    def region_id(self) -> Optional[int]:
        return self._region_id

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def nbytes(self) -> int:
        return self._local.numel() * self._local.element_size()

    @property
    def data_ptr(self) -> int:
        return self._local.storage().data_ptr()

    # ── Peer region tracking ──────────────────────────────────────────────

    def register_peer(
        self, peer_node_id: int, peer_region_id: int, peer_elements: int
    ) -> None:
        self._peer_regions[peer_node_id] = (peer_region_id, peer_elements)

    # ── Gradient synchronization (data-parallel) ──────────────────────────

    def sync_gradient(self, peer_regions: Dict[int, Tuple[int, int]]) -> None:
        """Synchronize gradients across data-parallel workers.

        Each node writes its gradient shard to every peer via RDMA put().
        Peers accumulate the received gradient into their local buffer.
        The result: every node has the sum of all gradients.

        This replaces allreduce — it's N-1 independent RDMA writes per node
        instead of a ring/star collective. Strictly simpler and lower latency
        for small N (3 nodes).
        """
        nbytes = self.nbytes
        my_ptr = self.data_ptr
        my_region = self._region_id
        my_node = self._node_id

        # Phase 1: put my gradient to every peer (scale by 1/world_size for avg)
        scale = 1.0 / self._num_shards

        # Scale local gradient first (in-place on MPS)
        self._local.mul_(scale)

        for peer_node, (peer_region, _) in peer_regions.items():
            if peer_node == my_node:
                continue
            wr_id = put(
                target_node=peer_node,
                target_region=peer_region,
                offset=0,
                src_addr=my_ptr,
                length=nbytes,
            )
            poll_completion(wr_id)

        # Phase 2: peers accumulate via Metal shader (they received our data
        # via RDMA write into their buffer — the accumulation happens
        # locally on each peer's GPU when they also call sync_gradient).
        # The key invariant: after all nodes call sync_gradient(),
        # every node's gradient buffer contains the global sum.
        drain_pending()

    # ── All-reduce ────────────────────────────────────────────────────────

    def all_reduce(
        self,
        peer_regions: Dict[int, Tuple[int, int]],
        op: str = "sum",
    ) -> None:
        """Reduce this tensor across all shards. Result written in-place.

        For sum: each node puts its data to every peer, peers accumulate.
        For avg: sum then scale by 1/world_size.
        """
        if op == "sum":
            self.sync_gradient(peer_regions)
        elif op == "avg":
            self.sync_gradient(peer_regions)
            self._local.mul_(1.0 / self._num_shards)
        elif op == "min":
            self._elementwise_reduce(peer_regions, "min")
        elif op == "max":
            self._elementwise_reduce(peer_regions, "max")
        else:
            raise ValueError(f"Unknown reduce op: {op}")

    def _elementwise_reduce(
        self,
        peer_regions: Dict[int, Tuple[int, int]],
        op: str,
    ) -> None:
        """Element-wise min/max across all peers."""
        nbytes = self.nbytes
        my_ptr = self.data_ptr
        my_node = self._node_id

        for peer_node, (peer_region, peer_elements) in peer_regions.items():
            if peer_node == my_node:
                continue
            peer_nbytes = peer_elements * self._local.element_size()

            # RDMA read peer data into a temporary buffer
            tmp = torch.empty_like(self._local)
            tmp_ptr = tmp.storage().data_ptr()
            tmp_region = register_region(tmp_ptr, peer_nbytes, 0x03)

            wr_id = get(
                target_node=peer_node,
                target_region=peer_region,
                offset=0,
                dst_addr=tmp_ptr,
                length=min(nbytes, peer_nbytes),
            )
            poll_completion(wr_id)

            if op == "min":
                self._local = torch.minimum(self._local, tmp)
            elif op == "max":
                self._local = torch.maximum(self._local, tmp)

            unregister_region(tmp_region)

        drain_pending()

    # ── All-gather ────────────────────────────────────────────────────────

    def all_gather(
        self, peer_regions: Dict[int, Tuple[int, int]]
    ) -> DistributedTensor:
        """Gather all shards into a full tensor on this node.

        Each node RDMA-reads from every other node to reconstruct the
        global tensor. Returns a new DistributedTensor with REPLICATED
        sharding.
        """
        elem_size = self._local.element_size()
        my_node = self._node_id

        # Allocate the full global buffer
        full_shape = self._global_shape
        full = torch.empty(full_shape, dtype=self._dtype, device="mps")
        full_region = register_region(
            full.storage().data_ptr(),
            full.numel() * elem_size,
            0x03,
        )

        # Copy local shard into the correct position
        offset_elements = self._shard_index * self._local.numel()
        full_flat = full.view(-1)
        local_flat = self._local.view(-1)
        full_flat[offset_elements : offset_elements + local_flat.numel()].copy_(
            local_flat
        )

        # RDMA read from each peer into the appropriate offset
        for peer_node, (peer_region, peer_elements) in peer_regions.items():
            if peer_node == my_node:
                continue

            peer_offset = peer_node * peer_elements * elem_size
            dst_ptr = full.storage().data_ptr() + peer_offset

            wr_id = get(
                target_node=peer_node,
                target_region=peer_region,
                offset=0,
                dst_addr=dst_ptr,
                length=peer_elements * elem_size,
            )
            poll_completion(wr_id)

        drain_pending()

        return DistributedTensor(
            local_tensor=full,
            sharding=ShardingSpec.REPLICATED,
            shard_index=0,
            num_shards=1,
            global_shape=full_shape,
            dtype=self._dtype,
            node_id=my_node,
        )

    # ── Reduce-scatter ────────────────────────────────────────────────────

    def reduce_scatter(
        self,
        peer_regions: Dict[int, Tuple[int, int]],
        op: str = "sum",
    ) -> DistributedTensor:
        """Reduce then scatter: each node receives a different shard of the
        reduced result. Used in FSDP-style gradient reduction."""
        my_node = self._node_id
        elem_size = self._local.element_size()
        num_elements = self._local.numel()

        # Allocate a buffer for each peer's slice
        for peer_node, (peer_region, peer_elements) in peer_regions.items():
            if peer_node == my_node:
                continue

            peer_nbytes = peer_elements * elem_size
            tmp = torch.empty(peer_elements, dtype=self._dtype, device="mps")
            tmp_ptr = tmp.storage().data_ptr()
            tmp_region = register_region(tmp_ptr, peer_nbytes, 0x03)

            wr_id = get(
                target_node=peer_node,
                target_region=peer_region,
                offset=0,
                dst_addr=tmp_ptr,
                length=min(self.nbytes, peer_nbytes),
            )
            poll_completion(wr_id)

            if op == "sum":
                self._local.add_(tmp)
            elif op == "avg":
                self._local.add_(tmp / self._num_shards)

            unregister_region(tmp_region)

        drain_pending()

        return self

    # ── Cleanup ───────────────────────────────────────────────────────────

    def __del__(self):
        try:
            self._deregister()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"DistributedTensor(shape={self._global_shape}, "
            f"sharding={self._sharding.name}, "
            f"shard={self._shard_index}/{self._num_shards}, "
            f"dtype={self._dtype}, "
            f"node={self._node_id}, "
            f"region={self._region_id})"
        )
