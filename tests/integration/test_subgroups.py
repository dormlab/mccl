import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES


def _worker(rank, world, sub_ranks):
    sub = dist.new_group(ranks=list(sub_ranks))
    x = torch.full((32,), float(rank + 100))
    if rank in sub_ranks:
        dist.all_reduce(x, group=sub)
    return x.cpu()


@pytest.mark.parametrize("world_size", [w for w in WORLD_SIZES if w >= 3])
def test_subgroup_allreduce(spawn, world_size):
    sub_ranks = (0, 1)
    res = spawn(world_size, _worker, sub_ranks=sub_ranks)
    in_sum = sum(r + 100 for r in sub_ranks)
    for rank, got in res.items():
        if rank in sub_ranks:
            assert torch.allclose(got, torch.full((32,), float(in_sum))), \
                f"in-group rank {rank}: got {got[0].item()}"
        else:
            assert torch.allclose(got, torch.full((32,), float(rank + 100))), \
                f"out-of-group rank {rank} was modified"


def _worker_2d(rank, world, dp_size, tp_size):
    dp_id = rank // tp_size
    tp_id = rank % tp_size
    dp_group = dist.new_group(ranks=[dp_id * tp_size + t for t in range(tp_size)])
    tp_group = dist.new_group(ranks=[d * tp_size + tp_id for d in range(dp_size)])
    x = torch.full((16,), float(rank))
    dist.all_reduce(x, group=tp_group)
    return x.cpu()


@pytest.mark.parametrize("dp_size,tp_size", [(2, 2)])
def test_2d_mesh(spawn, dp_size, tp_size):
    world = dp_size * tp_size
    res = spawn(world, _worker_2d, dp_size=dp_size, tp_size=tp_size)
    for rank, got in res.items():
        tp_id = rank % tp_size
        expected_sum = sum(d * tp_size + tp_id for d in range(dp_size))
        assert torch.allclose(got, torch.full((16,), float(expected_sum)))
