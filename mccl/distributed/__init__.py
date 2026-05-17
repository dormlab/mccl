"""
Distributed tensor and model-sharding runtime.
"""

from mccl.distributed.tensor import DistributedTensor, ShardingSpec
from mccl.distributed.sharding import DistributedModel, ModelParallelPlan

__all__ = [
    "DistributedTensor",
    "DistributedModel",
    "ShardingSpec",
    "ModelParallelPlan",
]
