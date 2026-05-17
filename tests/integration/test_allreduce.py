import pytest
import torch
import torch.distributed as dist

from conftest import (
    WORLD_SIZES, ALL_DTYPES, FLOAT_DTYPES,
    REDUCE_OPS_ALL, REDUCE_OPS_FLOAT_ONLY,
)


def _worker(rank, world, dtype, op):
    x = torch.full((128,), rank + 1, dtype=dtype)
    dist.all_reduce(x, op=op)
    return x.cpu()


def _per_rank_inputs(world, dtype):
    return [torch.full((128,), r + 1, dtype=dtype) for r in range(world)]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("op", REDUCE_OPS_ALL)
def test_allreduce(spawn, golden, world_size, dtype, op):
    if op == dist.ReduceOp.PRODUCT and dtype in (torch.int8, torch.uint8):
        pytest.skip("PRODUCT overflows on narrow int dtypes")
    res = spawn(world_size, _worker, dtype=dtype, op=op)
    expected = golden.allreduce(_per_rank_inputs(world_size, dtype), op).float()
    for rank, got in res.items():
        assert torch.allclose(got.float(), expected, atol=1e-2), \
            f"rank {rank}: got {got[0].item()}, expected {expected[0].item()}"


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("op", REDUCE_OPS_FLOAT_ONLY)
def test_allreduce_float_only_ops(spawn, golden, world_size, dtype, op):
    res = spawn(world_size, _worker, dtype=dtype, op=op)
    expected = golden.allreduce(_per_rank_inputs(world_size, dtype), op).float()
    for rank, got in res.items():
        assert torch.allclose(got.float(), expected, atol=1e-2)
