"""Pytest harness for the mccl backend.

Two helpers and four golden oracles, exposed as fixtures so any test
function can pull them with `def test_x(spawn, golden, ...)`.

To extend:
- new world size: append to WORLD_SIZES
- new dtype: append to FLOAT_DTYPES / INT_DTYPES
- new collective: write a module-level worker fn, parametrize on
  world_size and dtype, call spawn(world, worker, **kw), compare to
  golden.<collective>(...). ~20 LOC per file under tests/integration/.
"""
from __future__ import annotations

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZES = [int(x) for x in os.environ.get("MCCL_TEST_WORLDS", "2,3").split(",") if x.strip()]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
INT_DTYPES = [torch.int32, torch.int64, torch.uint8, torch.int8]
ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES

REDUCE_OPS_ALL = [
    dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.PRODUCT,
]
REDUCE_OPS_FLOAT_ONLY = [dist.ReduceOp.AVG]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _entry(rank, world, port, fn, kwargs, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        import mccl  # noqa: F401
        dist.init_process_group(backend="mccl", rank=rank, world_size=world)
        results[rank] = fn(rank, world, **kwargs)
    except Exception as e:
        results[rank] = ("__EXC__", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _spawn(world, fn, timeout: float = 60.0, **kwargs):
    """Run `fn(rank, world, **kwargs)` in `world` worker processes.

    Returns a dict {rank: return_value}. Any worker exception is surfaced
    as a pytest failure.
    """
    port = _free_port()
    mgr = mp.Manager()
    results = mgr.dict()
    ctx = mp.spawn(_entry, args=(world, port, fn, kwargs, results),
                   nprocs=world, join=False)
    if not ctx.join(timeout=timeout):
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        pytest.fail(f"spawn(world={world}) timed out after {timeout}s")
    out = dict(results)
    for r in range(world):
        if isinstance(out.get(r), tuple) and out[r] and out[r][0] == "__EXC__":
            pytest.fail(f"rank {r}: {out[r][1]}")
    return out


def _g_allreduce(per_rank_inputs, op):
    s = torch.stack([t.to(torch.float64) for t in per_rank_inputs])
    if op == dist.ReduceOp.SUM:     return s.sum(0)
    if op == dist.ReduceOp.AVG:     return s.mean(0)
    if op == dist.ReduceOp.MIN:     return s.min(0).values
    if op == dist.ReduceOp.MAX:     return s.max(0).values
    if op == dist.ReduceOp.PRODUCT: return s.prod(0)
    raise ValueError(f"unsupported op {op}")


def _g_broadcast(root_input):
    return root_input.to(torch.float64)


def _g_allgather(per_rank_inputs):
    return torch.cat([t.to(torch.float64) for t in per_rank_inputs], dim=0)


def _g_reduce_scatter(per_rank_inputs, op, rank, world):
    s = torch.stack([t.to(torch.float64) for t in per_rank_inputs])
    if op == dist.ReduceOp.SUM:     reduced = s.sum(0)
    elif op == dist.ReduceOp.AVG:   reduced = s.mean(0)
    elif op == dist.ReduceOp.MIN:   reduced = s.min(0).values
    elif op == dist.ReduceOp.MAX:   reduced = s.max(0).values
    elif op == dist.ReduceOp.PRODUCT: reduced = s.prod(0)
    else: raise ValueError(op)
    chunk = reduced.shape[0] // world
    return reduced[rank * chunk : (rank + 1) * chunk]


def _g_alltoall(per_rank_inputs, rank, world):
    chunk = per_rank_inputs[0].shape[0] // world
    return torch.cat([per_rank_inputs[p][rank * chunk : (rank + 1) * chunk].to(torch.float64)
                      for p in range(world)])


class _Golden:
    allreduce = staticmethod(_g_allreduce)
    broadcast = staticmethod(_g_broadcast)
    allgather = staticmethod(_g_allgather)
    reduce_scatter = staticmethod(_g_reduce_scatter)
    alltoall = staticmethod(_g_alltoall)


@pytest.fixture
def spawn():
    return _spawn


@pytest.fixture
def golden():
    return _Golden
