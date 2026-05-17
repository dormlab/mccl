import argparse
import os
import time

import torch
import torch.distributed as dist

import mccl  # noqa: F401


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

    n = 64
    ag_in = torch.full((n,), float(rank + 1), device=dev)
    ag_out = torch.empty(n * world_size, device=dev)
    dist.all_gather_into_tensor(ag_out, ag_in)
    expected_ag = torch.cat([torch.full((n,), float(p + 1)) for p in range(world_size)])
    assert torch.allclose(ag_out.cpu(), expected_ag), \
        f"all_gather_into_tensor mismatch on rank {rank}"

    rs_in = torch.cat([torch.full((n,), float((p + 1) * 10 + rank)) for p in range(world_size)]).to(dev)
    rs_out = torch.empty(n, device=dev)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    expected_rs = sum(torch.full((n,), float((rank + 1) * 10 + r)) for r in range(world_size))
    assert torch.allclose(rs_out.cpu(), expected_rs), \
        f"reduce_scatter_tensor mismatch on rank {rank}: got {rs_out[0].item()} expected {expected_rs[0].item()}"

    a2a_in = torch.tensor([rank * 10 + p for p in range(world_size)], dtype=torch.float32, device=dev)
    a2a_out = torch.empty(world_size, dtype=torch.float32, device=dev)
    dist.all_to_all_single(a2a_out, a2a_in)
    expected_a2a = torch.tensor([p * 10 + rank for p in range(world_size)], dtype=torch.float32)
    assert torch.allclose(a2a_out.cpu(), expected_a2a), \
        f"all_to_all_single mismatch on rank {rank}: got {a2a_out.tolist()} expected {expected_a2a.tolist()}"

    if world_size >= 2:
        sub_ranks = [0, 1]
        sub = dist.new_group(ranks=sub_ranks)
        sg_t = torch.full((16,), float(rank + 100), device=dev)
        if rank in sub_ranks:
            dist.all_reduce(sg_t, group=sub)
            expected_sub = torch.full((16,), float(sum(r + 100 for r in sub_ranks)))
            assert torch.allclose(sg_t.cpu(), expected_sub), \
                f"subgroup allreduce mismatch on rank {rank}: got {sg_t[0].item()}"
        else:
            assert torch.allclose(sg_t.cpu(), torch.full((16,), float(rank + 100))), \
                f"rank {rank} should be untouched by subgroup"

    if world_size > 1:
        next_r = (rank + 1) % world_size
        prev_r = (rank - 1) % world_size
        out_t = torch.full((n,), float(rank * 100), device=dev)
        in_t = torch.empty(n, device=dev)
        if rank % 2 == 0:
            dist.send(out_t, dst=next_r, tag=7)
            dist.recv(in_t, src=prev_r, tag=7)
        else:
            dist.recv(in_t, src=prev_r, tag=7)
            dist.send(out_t, dst=next_r, tag=7)
        expected_p2p = torch.full((n,), float(prev_r * 100))
        assert torch.allclose(in_t.cpu(), expected_p2p), \
            f"send/recv mismatch on rank {rank}: got {in_t[0].item()} expected {expected_p2p[0].item()}"

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
