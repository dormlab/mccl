import pytest
import torch
import torch.distributed as dist

from common import DTYPES, WORLD, assert_close


CHUNK = 8


def _per_rank_input(rank, world, dtype, device="mps"):
    return torch.cat([
        torch.full((CHUNK,), float(rank * 100 + p), dtype=dtype, device=device)
        for p in range(world)
    ])


def _worker_base(rank, world, dtype, in_place):
    inp = _per_rank_input(rank, world, dtype)
    if in_place:
        # In-place alltoall (output == input buffer)
        dist.all_to_all_single(inp, inp)
        return inp.cpu()
    out = torch.empty(CHUNK * world, dtype=dtype, device="mps")
    dist.all_to_all_single(out, inp)
    return out.cpu()


def _worker_uneven(rank, world, dtype):
    in_sizes = [4 * (rank + 1) for _ in range(world)]
    out_sizes = [4 * (p + 1) for p in range(world)]
    inp = torch.cat([
        torch.full((in_sizes[p],), float(rank * 100 + p), dtype=dtype, device="mps")
        for p in range(world)
    ])
    out = torch.empty(sum(out_sizes), dtype=dtype, device="mps")
    dist.all_to_all_single(out, inp, out_sizes, in_sizes)
    return out.cpu()


def _expected_even(rank, world, dtype):
    return torch.cat([
        torch.full((CHUNK,), float(p * 100 + rank), dtype=torch.float64)
        for p in range(world)
    ])


def _expected_uneven(rank, world, dtype):
    # chunk from peer p has length 4*(p+1) (peer p's in_sizes[*])
    return torch.cat([
        torch.full((4 * (p + 1),), float(p * 100 + rank), dtype=torch.float64)
        for p in range(world)
    ])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [False, True])
def test_alltoall_even(spawn, dtype, in_place):
    if dtype == torch.bool:
        pytest.skip("bool semantics not meaningful for arithmetic test values")
    res = spawn(_worker_base, dtype=dtype, in_place=in_place)
    for rank, got in res.items():
        assert_close(got, _expected_even(rank, WORLD, dtype), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_alltoall_uneven(spawn, dtype):
    if dtype == torch.bool:
        pytest.skip("bool semantics not meaningful for arithmetic test values")
    res = spawn(_worker_uneven, dtype=dtype)
    for rank, got in res.items():
        assert_close(got, _expected_uneven(rank, WORLD, dtype), dtype)
