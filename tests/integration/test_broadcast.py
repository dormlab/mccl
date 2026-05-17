import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES, ALL_DTYPES


def _worker(rank, world, dtype, root):
    if rank == root:
        x = torch.full((64,), 42, dtype=dtype)
    else:
        x = torch.zeros(64, dtype=dtype)
    dist.broadcast(x, src=root)
    return x.cpu()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_broadcast_root_0(spawn, golden, world_size, dtype):
    res = spawn(world_size, _worker, dtype=dtype, root=0)
    expected = golden.broadcast(torch.full((64,), 42, dtype=dtype)).float()
    for rank, got in res.items():
        assert torch.allclose(got.float(), expected)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_broadcast_root_last(spawn, golden, world_size):
    res = spawn(world_size, _worker, dtype=torch.float32, root=world_size - 1)
    expected = torch.full((64,), 42.0)
    for rank, got in res.items():
        assert torch.allclose(got, expected)
