import pytest
import torch
import torch.distributed as dist

from common import DTYPES, WORLD, seed_tensor, golden_allgather, assert_close


CHUNK = 32


def _worker_base(rank, world, dtype, in_place):
    inp = seed_tensor(rank, (CHUNK,), dtype)
    if in_place:
        full = torch.empty(CHUNK * world, dtype=dtype, device="mps")
        full.narrow(0, rank * CHUNK, CHUNK).copy_(inp)
        view_in = full.narrow(0, rank * CHUNK, CHUNK)
        dist.all_gather_into_tensor(full, view_in)
        return full.cpu()
    out = torch.empty(CHUNK * world, dtype=dtype, device="mps")
    dist.all_gather_into_tensor(out, inp)
    return out.cpu()


def _worker_list(rank, world, dtype):
    inp = seed_tensor(rank, (CHUNK,), dtype)
    outs = [torch.empty(CHUNK, dtype=dtype, device="mps") for _ in range(world)]
    dist.all_gather(outs, inp)
    return [o.cpu() for o in outs]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [False, True])
def test_allgather_base(spawn, dtype, in_place):
    res = spawn(_worker_base, dtype=dtype, in_place=in_place)
    inputs = [seed_tensor(r, (CHUNK,), dtype, device="cpu") for r in range(WORLD)]
    expected = golden_allgather(inputs)
    for _, got in res.items():
        assert_close(got, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_allgather_list(spawn, dtype):
    res = spawn(_worker_list, dtype=dtype)
    inputs = [seed_tensor(r, (CHUNK,), dtype, device="cpu") for r in range(WORLD)]
    for _, outs in res.items():
        for p in range(WORLD):
            assert_close(outs[p], inputs[p], dtype)
