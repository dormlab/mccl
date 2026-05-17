import pytest
import torch
import torch.distributed as dist

from conftest import WORLD_SIZES


def _worker_ring(rank, world, tag):
    nxt = (rank + 1) % world
    prv = (rank - 1) % world
    send_t = torch.full((32,), float(rank * 100))
    recv_t = torch.empty(32)
    if rank % 2 == 0:
        dist.send(send_t, dst=nxt, tag=tag); dist.recv(recv_t, src=prv, tag=tag)
    else:
        dist.recv(recv_t, src=prv, tag=tag); dist.send(send_t, dst=nxt, tag=tag)
    return recv_t.cpu()


@pytest.mark.parametrize("world_size", [w for w in WORLD_SIZES if w >= 2])
@pytest.mark.parametrize("tag", [0, 7, 12345])
def test_send_recv_ring(spawn, world_size, tag):
    res = spawn(world_size, _worker_ring, tag=tag)
    for rank, got in res.items():
        prv = (rank - 1) % world_size
        assert torch.equal(got, torch.full((32,), float(prv * 100))), \
            f"rank {rank}: got {got[0].item()} expected {float(prv * 100)}"


def _worker_wildcard(rank, world):
    nxt = (rank + 1) % world
    prv = (rank - 1) % world
    send_t = torch.full((16,), float(rank))
    recv_t = torch.empty(16)
    if rank % 2 == 0:
        dist.send(send_t, dst=nxt, tag=99); dist.recv(recv_t, src=prv, tag=-1)
    else:
        dist.recv(recv_t, src=prv, tag=-1); dist.send(send_t, dst=nxt, tag=99)
    return recv_t.cpu()


@pytest.mark.parametrize("world_size", [2, 3])
def test_send_recv_wildcard_tag(spawn, world_size):
    res = spawn(world_size, _worker_wildcard)
    for rank, got in res.items():
        prv = (rank - 1) % world_size
        assert torch.equal(got, torch.full((16,), float(prv)))
