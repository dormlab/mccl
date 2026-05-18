import pytest
import torch
import torch.distributed as dist

from common import (
    DTYPES, REDUCE_OPS, WORLD,
    is_reduce_dtype, seed_tensor, golden_allreduce, assert_close,
)


def _worker(rank, world, dtype, op):
    x = seed_tensor(rank, (128,), dtype)
    dist.all_reduce(x, op=op)
    return x.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", REDUCE_OPS)
def test_allreduce(spawn, dtype, op):
    if not is_reduce_dtype(dtype):
        pytest.skip("Metal reduce kernels currently cover float dtypes only")
    if op == dist.ReduceOp.PRODUCT and dtype == torch.float16:
        pytest.skip("PRODUCT in fp16 overflows quickly")
    res = spawn(_worker, dtype=dtype, op=op)
    inputs = [seed_tensor(r, (128,), dtype, device="cpu") for r in range(WORLD)]
    expected = golden_allreduce(inputs, op)
    for _, got in res.items():
        assert_close(got, expected, dtype)
