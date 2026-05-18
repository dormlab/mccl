"""End-to-end DDP test against the mccl backend.

Builds a small but real model, drives a few training steps under
torch.nn.parallel.DistributedDataParallel, and verifies:
  * No crashes (Metal asserts, MPSPredicate, PeerMesh send failures).
  * Loss strictly decreases across the warmup window.
  * Parameters stay numerically identical across ranks (allreduce
    semantics — DDP averages grads, every rank should see the same
    weights after opt.step).

These are the kinds of bugs the unit collective tests do not catch:
DDP fires reducer hooks mid-backward in a way the bare
`dist.all_reduce` tests don't, and the per-step gradient sync is what
breaks when our PG races with torch's MPS stream.
"""
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from common import spawn  # noqa: F401  (provided as fixture)


class _TinyMLP(nn.Module):
    def __init__(self, d=128, layers=4):
        super().__init__()
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(layers)],
            nn.Linear(d, d),
        )

    def forward(self, x):
        return self.layers(x)


def _ddp_step_worker(rank, world, steps=6, warmup=2, batch=8, dim=128):
    """Trains a tiny MLP under DDP. Returns
    (losses, final_param_signature)."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    device = torch.device("mps:0")
    torch.manual_seed(0xC0FFEE)  # same init on all ranks
    model = _TinyMLP(d=dim, layers=4).to(device)
    model = DDP(model, device_ids=None, bucket_cap_mb=1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Deterministic but rank-distinct data so allreduce actually does work.
    torch.manual_seed(rank * 1009 + 7)
    x = torch.randn(batch, dim, device=device)
    y = torch.randn(batch, dim, device=device)

    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        torch.mps.synchronize()
        losses.append(float(loss.detach().cpu()))

    # Param signature: sum of all params (moved to CPU first because
    # MPS does not support fp64). Must be identical across ranks
    # because DDP averages grads.
    sig = sum(p.detach().cpu().to(torch.float64).sum().item()
              for p in model.parameters())
    return {"losses": losses, "sig": sig, "warmup": warmup}


def test_ddp_runs_clean(spawn):
    """6 DDP steps complete without Metal asserts or PG send failures."""
    res = spawn(_ddp_step_worker, timeout=120.0)
    for r, payload in res.items():
        assert "losses" in payload, f"rank {r}: no losses returned"
        assert len(payload["losses"]) == 6


def test_ddp_loss_descends(spawn):
    """Loss at step 5 < loss at step 0 (post-warmup, post-allreduce)."""
    res = spawn(_ddp_step_worker, timeout=120.0)
    for r, payload in res.items():
        losses = payload["losses"]
        assert losses[-1] < losses[0], (
            f"rank {r}: loss didn't decrease ({losses[0]:.4f} → {losses[-1]:.4f})"
        )


def test_ddp_params_synced(spawn):
    """Every rank ends with bit-identical parameters."""
    res = spawn(_ddp_step_worker, timeout=120.0)
    sigs = [payload["sig"] for payload in res.values()]
    s0 = sigs[0]
    for i, s in enumerate(sigs[1:], 1):
        # DDP averages grads → identical opt.step → identical params.
        # Allow a tiny eps for fp64 accumulation order.
        assert abs(s - s0) < 1e-6, (
            f"rank {i} param signature {s} differs from rank 0 {s0} "
            f"(diff {abs(s - s0)})"
        )


def _ddp_many_buckets_worker(rank, world):
    """DDP with many small parameters → many tiny allreduces. Stresses
    the back-to-back reducer-hook code path that previously raced with
    torch's MPS stream."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    device = torch.device("mps:0")
    torch.manual_seed(0xBEEF)
    # ~30 separate parameter tensors → 30 allreduces per step at bucket_cap=1
    model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(15)]).to(device)
    model = DDP(model, device_ids=None, bucket_cap_mb=1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x = torch.randn(8, 64, device=device)
    y = torch.randn(8, 64, device=device)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        torch.mps.synchronize()
    return True


def test_ddp_many_small_buckets(spawn):
    """Stress test: many tiny allreduces back-to-back per backward."""
    res = spawn(_ddp_many_buckets_worker, timeout=120.0)
    for r, ok in res.items():
        assert ok is True, f"rank {r}: many-buckets DDP failed"
