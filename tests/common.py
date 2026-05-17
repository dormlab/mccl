"""Shared test harness for mccl collectives.

One source of truth — adding a new collective is a single file under
`tests/collectives/`. Add a new dtype = one line in DTYPES. Change the
world size at runtime via `MCCL_TEST_WORLD=N`.

Mirrors nccl-tests: each test seeds inputs deterministically, computes
the expected result in-process (no reference library), runs the
collective, and compares.
"""
from __future__ import annotations

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ── Runtime config ────────────────────────────────────────────────────

WORLD = int(os.environ.get("MCCL_TEST_WORLD", "2"))

# Dtypes torch+MPS+Metal can move and reduce. Reduce kernels currently
# cover floating point only; transport (broadcast/allgather/alltoall/
# send/recv) works for every dtype here. Tests that need reduce skip
# non-floating dtypes.
DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
]

REDUCE_OPS = [
    dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX,
    dist.ReduceOp.PRODUCT, dist.ReduceOp.AVG,
]


def is_reduce_dtype(d: torch.dtype) -> bool:
    return d.is_floating_point


def tol(d: torch.dtype) -> dict:
    if d in (torch.float16, torch.bfloat16): return {"rtol": 1e-2, "atol": 1e-2}
    if d == torch.float32: return {"rtol": 1e-4, "atol": 1e-4}
    return {"rtol": 0, "atol": 0}  # integers / bool: exact


# ── Worker spawning ───────────────────────────────────────────────────

def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


def _entry(rank, world, port, fn, kwargs, result_path):
    import pickle
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    payload = ("__EXC__", "worker did not complete")
    try:
        import mccl  # noqa: F401
        dist.init_process_group(
            backend="mccl", rank=rank, world_size=world,
            device_id=torch.device("mps:0"),
        )
        payload = ("OK", fn(rank, world, **kwargs))
    except Exception as e:
        payload = ("__EXC__",
                   f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")
    finally:
        try:
            with open(f"{result_path}/{rank}.pkl", "wb") as f:
                pickle.dump(payload, f)
        except Exception:
            pass
        try: dist.destroy_process_group()
        except Exception: pass
        # Force process exit to avoid hangs during interpreter teardown of
        # MPS/Progress state in subprocesses.
        os._exit(0)


def spawn(fn, *, world: int = None, timeout: float = 60.0, **kwargs):
    """Run fn(rank, world, **kwargs) on `world` workers (default WORLD).

    Returns {rank: result}. Any worker exception is a pytest failure.
    """
    import pickle, tempfile, shutil
    world = world or WORLD
    port = _free_port()
    result_dir = tempfile.mkdtemp(prefix="mccl_spawn_")
    try:
        ctx = mp.spawn(_entry, args=(world, port, fn, kwargs, result_dir),
                       nprocs=world, join=False)
        ok = False
        deadline = __import__("time").monotonic() + timeout
        while not ok and __import__("time").monotonic() < deadline:
            try:
                ok = ctx.join(timeout=1)
            except Exception as e:
                pytest.fail(f"spawn(world={world}) worker died: {e}")
        if not ok:
            for p in ctx.processes:
                if p.is_alive(): p.terminate()
            pytest.fail(f"spawn(world={world}) timed out after {timeout}s")
        out = {}
        for r in range(world):
            try:
                with open(f"{result_dir}/{r}.pkl", "rb") as f:
                    out[r] = pickle.load(f)
            except Exception as e:
                pytest.fail(f"rank {r}: result file missing ({e})")
        for r in range(world):
            tag, val = out[r]
            if tag == "__EXC__":
                pytest.fail(f"rank {r}: {val}")
        return {r: out[r][1] for r in range(world)}
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


# ── Deterministic input generation ────────────────────────────────────

def seed_tensor(rank: int, shape, dtype: torch.dtype, device="mps") -> torch.Tensor:
    """Predictable, dtype-safe input. Matches the golden reductions below."""
    n = 1
    for s in shape: n *= s
    if dtype == torch.bool:
        return torch.tensor([(rank + i) % 2 == 0 for i in range(n)],
                            dtype=torch.bool, device=device).view(*shape)
    if dtype in (torch.int8, torch.uint8):
        return torch.full(shape, (rank + 1) % 8, dtype=dtype, device=device)
    if not dtype.is_floating_point:
        return torch.full(shape, rank + 1, dtype=dtype, device=device)
    return torch.full(shape, float(rank + 1), dtype=dtype, device=device)


# ── Golden oracles (computed in-process, no reference library) ────────

def _stack64(inputs):
    return torch.stack([t.to(torch.float64).cpu() for t in inputs])


def golden_allreduce(inputs, op):
    s = _stack64(inputs)
    if op == dist.ReduceOp.SUM:     return s.sum(0)
    if op == dist.ReduceOp.AVG:     return s.mean(0)
    if op == dist.ReduceOp.MIN:     return s.min(0).values
    if op == dist.ReduceOp.MAX:     return s.max(0).values
    if op == dist.ReduceOp.PRODUCT: return s.prod(0)
    raise ValueError(op)


def golden_broadcast(root_input):
    return root_input.to(torch.float64).cpu()


def golden_allgather(inputs):
    return torch.cat([t.to(torch.float64).cpu() for t in inputs], dim=0)


def golden_reduce_scatter(inputs, op, rank, world):
    full = golden_allreduce(inputs, op)
    chunk = full.shape[0] // world
    return full[rank * chunk : (rank + 1) * chunk]


def golden_alltoall(inputs, rank, world):
    chunk = inputs[0].shape[0] // world
    return torch.cat([
        inputs[p][rank * chunk : (rank + 1) * chunk].to(torch.float64).cpu()
        for p in range(world)
    ])


def assert_close(got: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype):
    g = got.to(torch.float64).cpu()
    e = expected.to(torch.float64).cpu()
    assert torch.allclose(g, e, **tol(dtype)), \
        f"mismatch: got {g.flatten()[:4].tolist()} expected {e.flatten()[:4].tolist()}"
