import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES, ALL_DTYPES, REDUCE_OPS_ALL


def _worker_base(rank, world, dtype, op):
    chunks = [torch.full((32,), (p + 1) * 10 + rank, dtype=dtype) for p in range(world)]
    inp = torch.cat(chunks)
    out = torch.empty(32, dtype=dtype)
    dist.reduce_scatter_tensor(out, inp, op=op)
    return out.cpu()


def _per_rank_inputs(world, dtype):
    return [torch.cat([torch.full((32,), (p + 1) * 10 + r, dtype=dtype) for p in range(world)])
            for r in range(world)]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("op", REDUCE_OPS_ALL)
def test_reduce_scatter_tensor(spawn, golden, world_size, dtype, op):
    if op == dist.ReduceOp.PRODUCT and dtype in (torch.int8, torch.uint8):
        pytest.skip("PRODUCT overflows on narrow int dtypes")
    res = spawn(world_size, _worker_base, dtype=dtype, op=op)
    inputs = _per_rank_inputs(world_size, dtype)
    for rank, got in res.items():
        expected = golden.reduce_scatter(inputs, op, rank, world_size).float()
        assert torch.allclose(got.float(), expected, atol=1e-2), \
            f"rank {rank}: got {got[0].item()}, expected {expected[0].item()}"
