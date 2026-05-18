"""
Model parallelism plan: describes how a model is partitioned across the cluster.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import copy

import torch
import torch.nn as nn

from mccl._C import (
    register_region,
    unregister_region,
    put,
    get,
    poll_completion,
    drain_pending,
)
from mccl.distributed.tensor import DistributedTensor, ShardingSpec


class ModelParallelPlan:
    """Describes how a PyTorch model is sharded across the cluster.

    Three strategies:
      - Data parallel: each node gets a batch shard, full model replica
      - Tensor parallel: each node gets a column/row slice of weight matrices
      - Pipeline parallel: consecutive layers assigned to different nodes
    """

    def __init__(self, world_size: int, node_id: int = 0):
        self.world_size = world_size
        self.node_id = node_id

    def data_parallel_split(
        self, tensor: torch.Tensor, dim: int = 0
    ) -> List[DistributedTensor]:
        """Split a tensor along a dimension. Each node gets its shard."""
        shards = torch.chunk(tensor, self.world_size, dim=dim)
        return [
            DistributedTensor(
                local_tensor=shard.contiguous().to("mps"),
                sharding=ShardingSpec.DATA_PARALLEL,
                shard_index=i,
                num_shards=self.world_size,
                global_shape=tensor.shape,
                dtype=tensor.dtype,
            )
            for i, shard in enumerate(shards)
        ]

    def tensor_parallel_split(
        self, weight: torch.Tensor, dim: int = -1
    ) -> List[DistributedTensor]:
        """Split a weight matrix. Each node computes a portion of outputs."""
        shards = torch.chunk(weight, self.world_size, dim=dim)
        return [
            DistributedTensor(
                local_tensor=shard.contiguous().to("mps"),
                sharding=ShardingSpec.TENSOR_PARALLEL,
                shard_index=i,
                num_shards=self.world_size,
                global_shape=weight.shape,
                dtype=weight.dtype,
            )
            for i, shard in enumerate(shards)
        ]

    def pipeline_parallel_split(
        self, model_layers: List[nn.Module]
    ) -> Dict[int, List[nn.Module]]:
        """Assign consecutive layers to each pipeline stage.

        Returns {node_id: [list of layers]}.
        """
        layers_per_node = max(1, len(model_layers) // self.world_size)
        result: Dict[int, List[nn.Module]] = {}
        for i in range(self.world_size):
            start = i * layers_per_node
            end = (
                len(model_layers)
                if i == self.world_size - 1
                else start + layers_per_node
            )
            # Deep copy so each node has independent parameters
            result[i] = [copy.deepcopy(layer) for layer in model_layers[start:end]]
        return result


class DistributedModel:
    """Wraps a PyTorch model for distributed training.

    Each parameter is registered with DMEM as a SharedTensor, making
    gradients directly addressable by peer nodes via RDMA.

    Usage::

        model = MyModel().to("mps")
        dist_model = DistributedModel(model, node_id=0, world_size=3)
        dist_model.register_parameters()

        # Training loop
        for batch in dataloader:
            loss = dist_model(batch).mean()
            loss.backward()
            dist_model.sync_gradients()  # RDMA gradient sync, no allreduce
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        node_id: int = 0,
        world_size: int = 3,
    ):
        self.model = model
        self.node_id = node_id
        self.world_size = world_size
        self._param_regions: Dict[str, int] = {}  # param_name → region_id
        self._peer_param_regions: Dict[int, Dict[str, Tuple[int, int]]] = {}
        self._registered = False

    def register_parameters(self) -> None:
        """Register all model parameters with DMEM for RDMA access."""
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

            ptr = param.storage().data_ptr()
            nbytes = param.numel() * param.element_size()
            region_id = register_region(ptr, nbytes, 0x03)
            self._param_regions[name] = region_id

        self._registered = True

    def deregister_parameters(self) -> None:
        """Unregister all parameters from DMEM."""
        for name, region_id in self._param_regions.items():
            unregister_region(region_id)
        self._param_regions.clear()
        self._registered = False

    def set_peer_regions(
        self, peer_node_id: int, regions: Dict[str, Tuple[int, int]]
    ) -> None:
        """Register peer parameter regions for gradient sync.

        regions: {param_name: (region_id, num_elements)}
        """
        self._peer_param_regions[peer_node_id] = regions

    def sync_gradients(self) -> None:
        """Synchronize gradients across all nodes via RDMA writes.

        For each parameter, each node writes its gradient to every peer.
        After all nodes call this, everyone has the global gradient sum.
        """
        if not self._registered:
            raise RuntimeError("Call register_parameters() first")

        scale = 1.0 / self.world_size

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad
            if not grad.is_contiguous():
                grad = grad.contiguous()

            grad_ptr = grad.storage().data_ptr()
            grad_nbytes = grad.numel() * grad.element_size()
            my_region = self._param_regions[name]

            # Scale local gradient
            grad.mul_(scale)

            # RDMA write to each peer's gradient buffer
            for peer_node, peer_params in self._peer_param_regions.items():
                if peer_node == self.node_id:
                    continue

                peer_region, _ = peer_params.get(
                    name, (None, None)
                )
                if peer_region is None:
                    continue

                wr_id = put(
                    target_node=peer_node,
                    target_region=peer_region,
                    offset=0,
                    src_addr=grad_ptr,
                    length=grad_nbytes,
                )
                poll_completion(wr_id)

        drain_pending()

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __del__(self):
        try:
            self.deregister_parameters()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"DistributedModel(node={self.node_id}/{self.world_size}, "
            f"params={len(self._param_regions)}, "
            f"peers={list(self._peer_param_regions.keys())})"
        )
