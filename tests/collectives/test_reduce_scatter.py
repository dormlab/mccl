import pytest
import torch
import torch.distributed as dist

from common import (
    DTYPES, REDUCE_OPS, WORLD,
    is_reduce_dtype, golden_reduce_scatter, assert_close,
)


CHUNK = 32


def _per_rank_input(rank, world, dtype, device="mps"):
    return torch.cat([
        torch.full((CHUNK,), float((p + 1) * 10 + rank), dtype=dtype, device=device)
        for p in range(world)
    ])


def _worker_base(rank, world, dtype, op, in_place):
    inp = _per_rank_input(rank, world, dtype)
    if in_place:
        out = inp.narrow(0, rank * CHUNK, CHUNK)
        dist.reduce_scatter_tensor(out, inp, op=op)
        return out.contiguous().cpu()
    out = torch.empty(CHUNK, dtype=dtype, device="mps")
    dist.reduce_scatter_tensor(out, inp, op=op)
    return out.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", REDUCE_OPS)
@pytest.mark.parametrize("in_place", [False, True])
def test_reduce_scatter(spawn, dtype, op, in_place):
    if not is_reduce_dtype(dtype):
        pytest.skip("Metal reduce kernels currently cover float dtypes only")
    if op == dist.ReduceOp.PRODUCT and dtype == torch.float16:
        pytest.skip("PRODUCT in fp16 overflows")
    res = spawn(_worker_base, dtype=dtype, op=op, in_place=in_place)
    inputs = [_per_rank_input(r, WORLD, dtype, device="cpu") for r in range(WORLD)]
    for rank, got in res.items():
        expected = golden_reduce_scatter(inputs, op, rank, WORLD)
        assert_close(got, expected, dtype)
