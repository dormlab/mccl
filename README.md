# mccl

`mccl` is a `torch.distributed` backend for Apple Silicon clusters of Mac minis connected over Thunderbolt. It works with stock `torchrun` / `DistributedDataParallel`; the only change is `backend="mccl"`.

## Hardware

- **Thunderbolt cables between every pair of nodes.** mccl refuses non-TB interfaces by default (hard-filter on `MCCL_IFACE_PRIORITY` subnets). A single TB4 cable carries ~4.2 GB/s of TCP throughput.
- **Tailscale on each node** for control-plane / SSH. Data plane is TB only.
- Apple Silicon (arm64). Python ≥ 3.11. PyTorch ≥ 2.5 (2.12 tested).

## Setup

1. **Thunderbolt Bridge IPs.** Cable each pair of machines, then System Settings → Network → Thunderbolt Bridge → assign a `/24` per peer pair. The defaults `mccl` looks for are `192.168.101.0/24`, `192.168.102.0/24`, `192.168.103.0/24`.
2. **Xcode CLI tools**: `xcode-select --install`.
3. **Install**:
   ```bash
   pip install torch
   pip install -e .
   ```
4. **Verify**:
   ```bash
   mccl --check
   ```

## Usage

```python
import mccl                  # MUST import before torch (see Performance notes)
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="mccl", device_id=torch.device("mps:0"))
model = DDP(MyModel().to("mps:0"))
# ...train as usual
```

Launch per node:
```bash
RANK=$N WORLD_SIZE=3 MASTER_ADDR=<rank0-ip> MASTER_PORT=29500 python train.py
```

## Performance

Measured on a 3× Mac mini (M4, 16 GB) cluster, full TB triangle, **70M-param decoder-only transformer**, batch=16, seq=256, bf16-on-wire, butterfly allreduce.

| Config | Samples/s | Speedup |
|---|---|---|
| Single mini | 22.1 | 1.00× |
| 3 minis, ring allreduce | 54.1 | 2.44× |
| **3 minis, butterfly allreduce + bf16 wire** | **56.9** | **2.58×** |

Stable across 3 consecutive runs (variance < 1 %). The remaining gap to 3× is hardware-bound: each peer pair has one TB cable, capping comm at 4.2 GB/s; the theoretical ceiling is ~2.7×.

110M-param transformer (batch=16): single mini 14.8 → 3-mini 35.9 = **2.43×**.

## Tunables

| Env var | Default | Purpose |
|---|---|---|
| `MCCL_IFACE_PRIORITY` | `192.168.101.,192.168.102.,192.168.103.` | TB subnet priority. Hard-filtered — non-priority IPs refused. |
| `MCCL_ALLREDUCE_ALGO` | `auto` | `ring`, `tree`, `butterfly`, or `auto` (tree below 256 KB else ring). `butterfly` is fastest on full TB mesh, N ≤ 4. |
| `MCCL_TREE_BELOW` | `262144` | Byte threshold below which `auto` picks tree over ring. |
| `MCCL_WIRE_DTYPE` | unset | Set to `bf16` to compress fp32 wire payloads (halves bandwidth, marginal accuracy hit). |
| `PYTORCH_MPS_TRACE_SIGNPOSTS` | auto-set to `1` | Disables torch's `commitAndContinue` to avoid an MPS race during DDP backward. mccl sets this on import (must precede `import torch`). |

## Collectives

`allreduce` (ring / tree / butterfly) · `broadcast` · `barrier` · `allgather` · `reduce_scatter` · `alltoall` · `send` / `recv`

## Parallelism support

mccl exposes the c10d primitives every PyTorch parallelism strategy needs. Status of each at scale on 3-mini cluster:

- **Data parallel (DDP)** — verified end-to-end (2.58× at 70M, 2.43× at 110M).
- **Pipeline parallel** — verified via `dist.send`/`dist.recv` between stages.
- **Tensor parallel** — primitives verified; `torch.distributed.tensor.parallel.DeviceMesh("mps", …)` is broken in torch 2.12 (missing `torch.mps.is_initialized`), so hand-wired col/row parallel works but the high-level API does not.
- **FSDP / ZeRO-3** — primitives present (allgather + reduce_scatter pass unit tests) but torch's FSDP itself is **not yet wired to MPS** (`UntypedStorage.resize_: got unexpected device type mps`). This is an upstream torch limitation, not mccl.

## Upstream PyTorch patch

Optional but recommended: `patches/pytorch/0001-MPSStream-flush-defensive-status-check.patch`. Fixes a torch MPS race that mccl's async overlap path triggers under DDP. mccl ships with inline-sync allreduce so the patch isn't required for the steady 2.58× number above; apply it only if you want to experiment with async overlap.

## Internals

Apple Silicon **UMA** means CPU and GPU share physical RAM. `MTLBuffer` (shared storage) gives a CPU pointer that aliases GPU memory — `extract_mps_buffer` (`csrc/metal/MPSInterop.mm`) hands that pointer to `send(2)`/`recv(2)` directly, no staging copy.

mccl's ring & butterfly allreduce reduce in place into the tensor slice via Metal kernels (`csrc/metal/MetalKernels.mm`), drained on a per-PG MTL queue. The Progress engine handles other collectives async; allreduce is intentionally inline-sync to avoid a torch MPS commit race.

## License

MIT — [LICENSE](LICENSE)
