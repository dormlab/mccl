import pytest
import torch
import torch.distributed as dist


def _send_to_self(rank, world):
    try:
        dist.send(torch.zeros(4), dst=rank, tag=0)
        return "no-error"
    except Exception as e:
        return f"{type(e).__name__}"


def _recv_size_mismatch(rank, world):
    nxt = (rank + 1) % world
    prv = (rank - 1) % world
    send_t = torch.zeros(64)
    recv_t = torch.empty(32)
    try:
        if rank % 2 == 0:
            dist.send(send_t, dst=nxt, tag=1); dist.recv(recv_t, src=prv, tag=1)
        else:
            dist.recv(recv_t, src=prv, tag=1); dist.send(send_t, dst=nxt, tag=1)
        return "no-error"
    except Exception as e:
        return f"{type(e).__name__}"


@pytest.mark.parametrize("world_size", [2])
def test_send_to_self_errors(spawn, world_size):
    res = spawn(world_size, _send_to_self)
    for rank, got in res.items():
        assert got != "no-error", f"rank {rank} did not error on send-to-self"


@pytest.mark.parametrize("world_size", [2])
def test_recv_size_mismatch_errors(spawn, world_size):
    res = spawn(world_size, _recv_size_mismatch)
    assert any(v != "no-error" for v in res.values()), \
        "expected at least one rank to error on size mismatch"
