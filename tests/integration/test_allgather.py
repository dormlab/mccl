import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES, ALL_DTYPES


def _worker_base(rank, world, dtype):
    out = torch.empty(32 * world, dtype=dtype)
    inp = torch.full((32,), rank + 1, dtype=dtype)
    dist.all_gather_into_tensor(out, inp)
    return out.cpu()


def _per_rank_inputs(world, dtype):
    return [torch.full((32,), r + 1, dtype=dtype) for r in range(world)]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_allgather_into_tensor(spawn, golden, world_size, dtype):
    res = spawn(world_size, _worker_base, dtype=dtype)
    expected = golden.allgather(_per_rank_inputs(world_size, dtype)).float()
    for rank, got in res.items():
        assert torch.allclose(got.float(), expected)


def _worker_list(rank, world, dtype):
    outs = [torch.empty(32, dtype=dtype) for _ in range(world)]
    inp = torch.full((32,), rank + 1, dtype=dtype)
    dist.all_gather(outs, inp)
    return [t.cpu() for t in outs]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_allgather_list_form(spawn, golden, world_size):
    res = spawn(world_size, _worker_list, dtype=torch.float32)
    expected_per_peer = _per_rank_inputs(world_size, torch.float32)
    for rank, outs in res.items():
        for p, got in enumerate(outs):
            assert torch.equal(got, expected_per_peer[p]), \
                f"rank {rank} slot {p}: got {got[0]}, expected {expected_per_peer[p][0]}"
