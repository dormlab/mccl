import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES


def _worker_even(rank, world, dtype):
    inp = torch.tensor([rank * 100 + p for p in range(world)], dtype=dtype).repeat_interleave(8)
    out = torch.empty(world * 8, dtype=dtype)
    dist.all_to_all_single(out, inp)
    return out.cpu()


def _worker_uneven(rank, world, dtype):
    in_sizes  = [4 * (rank + 1) for _ in range(world)]
    out_sizes = [4 * (p + 1) for p in range(world)]
    inp = torch.cat([torch.full((in_sizes[p],), rank * 100 + p, dtype=dtype) for p in range(world)])
    out = torch.empty(sum(out_sizes), dtype=dtype)
    dist.all_to_all_single(out, inp, out_sizes, in_sizes)
    return out.cpu()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_alltoall_even(spawn, world_size, dtype):
    res = spawn(world_size, _worker_even, dtype=dtype)
    for rank, got in res.items():
        expected = torch.tensor([p * 100 + rank for p in range(world_size)], dtype=dtype).repeat_interleave(8)
        assert torch.equal(got, expected), \
            f"rank {rank}: got {got[:4].tolist()}, expected {expected[:4].tolist()}"


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_alltoall_uneven(spawn, world_size):
    res = spawn(world_size, _worker_uneven, dtype=torch.float32)
    for rank, got in res.items():
        expected_chunks = [torch.full((4 * (p + 1),), p * 100 + rank, dtype=torch.float32) for p in range(world_size)]
        expected = torch.cat(expected_chunks)
        assert torch.equal(got, expected), f"rank {rank}: mismatch"
