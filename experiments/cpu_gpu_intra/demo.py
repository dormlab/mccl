"""Prove that on Apple Silicon UMA, CPU and GPU can do useful per-step
training work simultaneously.

Setup:
    A ~50M decoder-only transformer on mps:0. Training loop runs 20 steps
    twice — once with the optimizer step fully on GPU, once with half the
    parameters' Aurora step offloaded to a CPU thread that runs in
    parallel with the GPU's next-batch forward.

Aurora is chosen because it's compute-heavy (8 matmuls per matrix via
Polar Express); an AdamW step is too cheap to make the overlap visible.

What we measure:
    - Wall-clock step time, mean and per-step
    - Per-core CPU% time series during each step
    - MPS allocator footprint (sanity)

If the parallel mode reduces step time AND per-core CPU% spikes above
~10% during what is mostly a GPU phase, we've demonstrated genuine
CPU+GPU concurrent execution on UMA.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn

# Aurora optimizer lives over in silicon_cluster — point at it directly.
SILICON = Path.home() / "silicon_cluster/src/run1/scrpts/optimizers"
sys.path.insert(0, str(SILICON))
from aurora import aurora as aurora_step  # noqa: E402
from polar_express import polar_express  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import Monitor  # noqa: E402


# --------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, d, bias=False),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyTx(nn.Module):
    """~50M dec-only, vocab=16K, d=512, layers=10."""
    def __init__(self, vocab=16000, d=512, nh=8, ff=2048, n_layers=10, seq=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))
        self.blocks = nn.ModuleList([Block(d, nh, ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, : idx.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------- aurora utilities

def collect_aurora_params(model):
    """2D matmul weights only (Muon convention)."""
    out = []
    for name, p in model.named_parameters():
        if p.dim() == 2 and "ln" not in name and "emb" not in name:
            out.append((name, p))
    return out


def make_cpu_mirrors(named_params):
    """For each MPS param, allocate a CPU twin used by the CPU Aurora thread.

    On Apple Silicon Shared MTLBuffer storage *is* host memory, but
    torch's MPS-> CPU `.to('cpu')` still goes through a copy in 2.12.
    We accept that small copy (a few MB / matrix) for this demo.
    Returns: name -> {"cpu_w": cpu_tensor, "cpu_g": cpu_tensor,
                       "cpu_m": cpu_tensor, "mps_w": mps_tensor}
    """
    mirrors = {}
    for name, p in named_params:
        mirrors[name] = {
            "mps_w": p,
            "cpu_w": p.detach().to("cpu", dtype=torch.float32).contiguous(),
            "cpu_g": torch.zeros_like(p, device="cpu", dtype=torch.float32),
            "cpu_m": torch.zeros_like(p, device="cpu", dtype=torch.float32),
        }
    return mirrors


def cpu_aurora_chunk(mirrors, names, eta, wd, mu):
    """Run aurora_step on CPU for a list of param names. Called from
    a worker thread; uses torch CPU ops (release GIL inside ATen)."""
    for name in names:
        m = mirrors[name]
        # We previously synced m["cpu_g"] from the GPU's gradient.
        aurora_step(
            m["cpu_w"], m["cpu_g"], m["cpu_m"],
            eta=eta, weight_decay=wd, mu=mu,
        )


def gpu_aurora_chunk(named_params, momenta_mps, names_set, eta, wd, mu):
    """Run aurora_step on MPS for the params whose name is in names_set."""
    for name, p in named_params:
        if name not in names_set or p.grad is None:
            continue
        aurora_step(
            p.data, p.grad, momenta_mps[name],
            eta=eta, weight_decay=wd, mu=mu,
        )


# --------------------------------------------------------------- mode

def run_mode(mode: str, n_steps: int, batch: int, seq: int, device: str,
             cpu_fraction: float, executor: ThreadPoolExecutor | None,
             seed: int = 42) -> dict:
    torch.manual_seed(seed)
    model = TinyTx(seq=seq).to(device)
    named = collect_aurora_params(model)
    n_params_total = sum(p.numel() for p in model.parameters())
    n_matmul = len(named)

    # Split matrices for CPU vs GPU. First `floor(cpu_fraction*N)` go to CPU.
    n_cpu = int(round(n_matmul * cpu_fraction)) if mode == "parallel" else 0
    cpu_names = {n for n, _ in named[:n_cpu]}
    gpu_names = {n for n, _ in named[n_cpu:]}

    # Momentum buffers per device.
    mirrors = make_cpu_mirrors([(n, p) for n, p in named if n in cpu_names])
    momenta_mps = {n: torch.zeros_like(p) for n, p in named if n in gpu_names}

    # Random "data".
    def get_batch():
        return (
            torch.randint(0, 16000, (batch, seq), device=device),
            torch.randint(0, 16000, (batch, seq), device=device),
        )

    loss_fn = nn.CrossEntropyLoss()
    losses = []
    step_ms = []
    # Pre-fire first batch so the first step isn't measuring data gen.
    x, y = get_batch()

    monitor = Monitor(interval=0.02)
    monitor.start()
    t0_all = time.perf_counter()

    cpu_future = None
    for step in range(n_steps):
        t0 = time.perf_counter()

        # If a CPU optimizer chunk from the previous step is still running,
        # wait for it before doing this step's optimizer (we need its
        # updated weights to be reflected on MPS before forward).
        if cpu_future is not None:
            cpu_future.result()
            # Copy CPU-updated weights back to MPS in-place.
            for name in cpu_names:
                m = mirrors[name]
                m["mps_w"].data.copy_(m["cpu_w"].to(device,
                                                     dtype=m["mps_w"].dtype))
            cpu_future = None

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, 16000), y.reshape(-1))
        loss.backward()

        # === The interesting part ===
        # In "parallel" mode: spawn CPU Aurora on cpu_names in a thread,
        # then run GPU Aurora on gpu_names on the main thread, then
        # kick off the next batch's forward immediately while CPU is still
        # running.
        # In "sequential" mode: run everything on GPU, no thread.
        if mode == "parallel" and executor is not None and cpu_names:
            # Push gradients to CPU mirrors (small copy per matrix).
            for name in cpu_names:
                m = mirrors[name]
                # MPS grads are bf16/fp32; cast to fp32 on CPU mirror.
                p = next(pp for nn, pp in named if nn == name)
                m["cpu_g"].copy_(p.grad.detach().to("cpu",
                                                     dtype=torch.float32))
            cpu_future = executor.submit(
                cpu_aurora_chunk, mirrors, list(cpu_names),
                0.05, 0.025, 0.95)
            # GPU side: only the GPU-assigned matrices.
            gpu_aurora_chunk(named, momenta_mps, gpu_names, 0.05, 0.025, 0.95)
            # Kick off next batch's data prep + forward setup so we can
            # let the GPU work overlap with CPU thread.
            x_next, y_next = get_batch()
        else:
            # All-GPU baseline.
            gpu_aurora_chunk(
                named, momenta_mps, {n for n, _ in named},
                0.05, 0.025, 0.95)
            x_next, y_next = get_batch()

        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        step_ms.append(dt * 1000.0)
        losses.append(float(loss.detach().cpu()))
        x, y = x_next, y_next

        print(f"  [{mode}] step {step:2d}  loss {losses[-1]:7.3f}  "
              f"step_ms={dt*1000:.0f}", flush=True)

    if cpu_future is not None:
        cpu_future.result()
    torch.mps.synchronize()
    monitor.stop()
    total = time.perf_counter() - t0_all

    return {
        "mode": mode,
        "n_params": n_params_total,
        "n_matmul": n_matmul,
        "cpu_matrices": n_cpu,
        "gpu_matrices": n_matmul - n_cpu,
        "steps": n_steps,
        "total_s": total,
        "median_step_ms": sorted(step_ms)[len(step_ms) // 2],
        "step_ms": step_ms,
        "losses": losses,
        "monitor": monitor.summary(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--cpu_fraction", type=float, default=0.5,
                   help="Fraction of matrices to offload to CPU in parallel mode.")
    p.add_argument("--out", type=str, default="/tmp/cpu_gpu_intra.json")
    args = p.parse_args()

    device = "mps:0"
    print(f"== running on {device} ==", flush=True)
    n_threads = torch.get_num_threads()
    print(f"== torch.get_num_threads() = {n_threads}", flush=True)
    print()

    # Warm up MPS once so the first measurement isn't biased.
    _ = TinyTx(seq=args.seq).to(device)
    torch.mps.synchronize()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-opt")
    print("== sequential (all GPU) ==", flush=True)
    seq_result = run_mode(
        "sequential", args.steps, args.batch, args.seq, device,
        cpu_fraction=0.0, executor=None, seed=42)
    print(f"  total {seq_result['total_s']:.2f}s  median {seq_result['median_step_ms']:.0f}ms",
          flush=True)
    print(f"  monitor: {seq_result['monitor']}", flush=True)
    print()

    print(f"== parallel (cpu_fraction={args.cpu_fraction}) ==", flush=True)
    par_result = run_mode(
        "parallel", args.steps, args.batch, args.seq, device,
        cpu_fraction=args.cpu_fraction, executor=executor, seed=42)
    print(f"  total {par_result['total_s']:.2f}s  median {par_result['median_step_ms']:.0f}ms",
          flush=True)
    print(f"  monitor: {par_result['monitor']}", flush=True)
    print()

    speedup = seq_result["median_step_ms"] / par_result["median_step_ms"]
    print("=" * 60, flush=True)
    print(f"SPEEDUP (median step):  {speedup:.2f}x", flush=True)
    seq_cpu = seq_result["monitor"]["proc_cpu_avg_pct"]
    par_cpu = par_result["monitor"]["proc_cpu_avg_pct"]
    print(f"Process CPU%:  seq {seq_cpu:.0f}%  ->  par {par_cpu:.0f}%",
          flush=True)
    print(f"CPU peak (per-core sum %): "
          f"seq {seq_result['monitor']['cpu_total_peak_pct']*seq_result['monitor']['ncpu']:.0f} "
          f"-> par {par_result['monitor']['cpu_total_peak_pct']*par_result['monitor']['ncpu']:.0f}",
          flush=True)
    print(f"GPU peak driver mem (MB): "
          f"seq {seq_result['monitor']['mps_driver_peak_mb']:.0f} "
          f"par {par_result['monitor']['mps_driver_peak_mb']:.0f}",
          flush=True)

    payload = {"sequential": seq_result, "parallel": par_result,
               "speedup": speedup}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nfull results saved to {args.out}", flush=True)

    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
