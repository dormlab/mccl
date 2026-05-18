# CPU + GPU intra-step concurrent compute on UMA — experiment

## Question

Apple Silicon has UMA: CPU and GPU share the same physical RAM. Can they
run useful training-step work **simultaneously** on the same tensors,
and does it help wall-clock time?

## Setup

`demo.py` runs the same ~50M decoder-only transformer twice on an M4 mini:

1. **Sequential**: Aurora optimizer step entirely on GPU (Metal).
2. **Parallel**: half of the 2D weight matrices' Aurora step is offloaded
   to a CPU `ThreadPoolExecutor` worker (torch CPU + Accelerate AMX),
   the other half stays on GPU, in parallel.

`monitor.py` samples per-core CPU%, process CPU%, and
`torch.mps.driver_allocated_memory()` every 20 ms throughout the run.

```
python experiments/cpu_gpu_intra/demo.py --steps 12 --cpu_fraction 0.5
```

## Result (lexie, M4 16GB, torch 2.12-patched)

| Metric | Sequential | Parallel |
|---|---|---|
| Median step | **760 ms** | **65,778 ms** |
| Process CPU avg | 13 % | **110 %** |
| Process CPU peak | 164 % | **297 %** |
| Per-core sum peak | 250 % | **725 %** |
| MPS driver peak | 2.32 GB | 2.33 GB |
| Loss 0→11 | 422 → 68 | 422 → 70 |

## Interpretation

- **Concurrent CPU+GPU execution on UMA works.** CPU usage tripled,
  GPU memory unchanged, both compute units physically active on the
  same RAM. No platform/driver issue.
- **Aurora's polar on this matrix size is the wrong workload.** AMX-fp32
  on a (512, 2048) matrix is ~50× slower than Metal-bf16 on the same
  matrix. Half-offloading dragged total step time to the CPU's pace.
- **Where the technique pays off**: workloads where per-op compute is
  small enough that kernel-launch overhead dominates (elementwise opt
  steps on small tensors, gradient clipping, dataloader work). The
  Aurora polar step on real model dims is not one of them.

## What we keep

- `demo.py` — a reusable harness to A/B any CPU↔GPU split for an
  optimizer step.
- `monitor.py` — psutil-backed sampler that records the time series
  of CPU/MPS usage during the run.

These are the right tools to answer "would offloading X to CPU help?"
in 30 seconds for any future workload.
