import argparse
import os
import time

import torch
import torch.distributed as dist

import distro  # noqa: F401


def run(rank: int, world_size: int, master: str, port: int, device: str):
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="mccl", rank=rank, world_size=world_size)

    dev = torch.device(device)

    x = torch.full((1024,), float(rank + 1), device=dev)
    dist.all_reduce(x)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(x.cpu(), torch.full((1024,), float(expected))), \
        f"allreduce mismatch on rank {rank}: got {x[0].item()} expected {expected}"

    b = torch.zeros(64, device=dev)
    if rank == 0:
        b.fill_(42.0)
    dist.broadcast(b, src=0)
    assert torch.allclose(b.cpu(), torch.full((64,), 42.0)), \
        f"broadcast mismatch on rank {rank}"

    if rank == 0:
        print(f"correctness ok (world={world_size}, device={device})")

    sizes = [1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22]
    iters = 10
    for nbytes in sizes:
        n = nbytes // 4
        t = torch.ones(n, device=dev)
        dist.barrier()
        torch.mps.synchronize() if device == "mps" else None
        s = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.mps.synchronize() if device == "mps" else None
        dt = (time.perf_counter() - s) / iters
        if rank == 0:
            bw = nbytes / dt / 1e6
            print(f"  allreduce  size={nbytes:>10d} B  t={dt*1000:7.2f} ms  bw={bw:7.1f} MB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    a = p.parse_args()
    run(a.rank, a.world, a.master, a.port, a.device)
