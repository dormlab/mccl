import pytest
import torch
import torch.distributed as dist

from common import DTYPES, WORLD, seed_tensor, assert_close


def _worker_ring(rank, world, dtype, tag):
    nxt = (rank + 1) % world
    prv = (rank - 1) % world
    send_t = seed_tensor(rank, (32,), dtype)
    recv_t = torch.empty(32, dtype=dtype, device="mps")
    if rank % 2 == 0:
        dist.send(send_t, dst=nxt, tag=tag); dist.recv(recv_t, src=prv, tag=tag)
    else:
        dist.recv(recv_t, src=prv, tag=tag); dist.send(send_t, dst=nxt, tag=tag)
    return recv_t.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tag", [0, 7, 12345])
def test_send_recv_ring(spawn, dtype, tag):
    res = spawn(_worker_ring, dtype=dtype, tag=tag)
    for rank, got in res.items():
        prv = (rank - 1) % WORLD
        expected = seed_tensor(prv, (32,), dtype, device="cpu").to(torch.float64)
        assert_close(got, expected, dtype)


def _worker_wildcard(rank, world, dtype):
    nxt = (rank + 1) % world
    prv = (rank - 1) % world
    send_t = seed_tensor(rank, (16,), dtype)
    recv_t = torch.empty(16, dtype=dtype, device="mps")
    if rank % 2 == 0:
        dist.send(send_t, dst=nxt, tag=42); dist.recv(recv_t, src=prv, tag=-1)
    else:
        dist.recv(recv_t, src=prv, tag=-1); dist.send(send_t, dst=nxt, tag=42)
    return recv_t.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
def test_send_recv_wildcard_tag(spawn, dtype):
    res = spawn(_worker_wildcard, dtype=dtype)
    for rank, got in res.items():
        prv = (rank - 1) % WORLD
        expected = seed_tensor(prv, (16,), dtype, device="cpu").to(torch.float64)
        assert_close(got, expected, dtype)
