import pytest
import torch
import torch.distributed as dist

from common import DTYPES, WORLD, seed_tensor, golden_broadcast, assert_close


def _worker(rank, world, dtype, root):
    if rank == root:
        x = seed_tensor(rank, (64,), dtype)
    else:
        x = torch.zeros(64, dtype=dtype, device="mps")
    dist.broadcast(x, src=root)
    return x.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("root", [0, "last"])
def test_broadcast(spawn, dtype, root):
    real_root = 0 if root == 0 else WORLD - 1
    res = spawn(_worker, dtype=dtype, root=real_root)
    expected = golden_broadcast(seed_tensor(real_root, (64,), dtype, device="cpu"))
    for _, got in res.items():
        assert_close(got, expected, dtype)
