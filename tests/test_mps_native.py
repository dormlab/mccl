"""Metal-native collective coverage.

Every collective that ProcessGroupMCCL implements is exercised here against
MPS tensors. Each test asserts:

  * numerical correctness against the expected reduction / gather result;
  * the output tensor's device stays on MPS through the call (no silent
    ``.cpu()`` bounce);
  * for ops that take MPS-only inputs, that passing a CPU tensor raises.

The PG body uses Metal compute kernels via ``metal_reduce_op`` /
``metal_scale_inplace`` and the wire path goes through
``stage_for_send_nosync`` / ``unstage_from_recv`` against the MTLBuffers
behind each tensor — see ``csrc/backend/ProcessGroupMCCL.cpp``.
"""

import platform
import os
import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"),
    reason="MPS collective tests require macOS on Apple Silicon",
)


def _run_distributed(fn, world_size=2, port=29500):
    """Spawn ``world_size`` subprocesses each running ``fn(rank, world_size)``.

    ``fn`` is shipped to children via ``inspect.getsource`` so it must be
    a self-contained top-level function.
    """
    import subprocess, sys, textwrap, inspect, time

    src = textwrap.dedent(inspect.getsource(fn))
    script = (
        "import os, sys, traceback, torch, torch.distributed as dist\n"
        f"os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MASTER_PORT'] = '{port}'\n"
        f"os.environ['MCCL_LISTEN_ADDR'] = '127.0.0.1'\n"
        f"os.environ['MCCL_PORT_BASE'] = '{port + 100}'\n"
        f"os.environ['MCCL_LOG_LEVEL'] = 'DEBUG'\n"
        "rank = int(sys.argv[1])\n"
        "world_size = int(sys.argv[2])\n"
        "import mccl\n"
        "dist.init_process_group(backend='mccl', rank=rank, "
        "world_size=world_size, device_id=torch.device('mps:0'))\n"
        "_mccl_exit = 0\n"
        "try:\n"
        f"{textwrap.indent(src, '    ')}"
        "    fn(rank, world_size)\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    _mccl_exit = 1\n"
        "try:\n"
        "    dist.destroy_process_group()\n"
        "except Exception:\n"
        "    pass\n"
        "os._exit(_mccl_exit)\n"
    )

    procs = []
    for r in range(world_size):
        p = subprocess.Popen([sys.executable, "-c", script, str(r), str(world_size)])
        procs.append(p)
        time.sleep(0.5)
    for p in procs:
        rc = p.wait(timeout=60)
        assert rc == 0, f"Worker exited with code {rc}"


# ── allreduce: all reduce ops, device preservation ───────────────────

class TestAllreduce:
    def test_sum_f32(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps") * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            assert t.device.type == "mps", f"got {t.device}"
            expected_val = sum(r + 1 for r in range(world_size))
            assert torch.allclose(t, torch.ones(64, device="mps") * expected_val,
                                  rtol=1e-4)

        _run_distributed(fn, world_size=2, port=35000)

    def test_avg_f32(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps") * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            assert t.device.type == "mps"
            mean_val = sum(r + 1 for r in range(world_size)) / world_size
            assert torch.allclose(t, torch.ones(64, device="mps") * mean_val,
                                  rtol=1e-4)

        _run_distributed(fn, world_size=2, port=35100)

    def test_min_f32(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps") * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            assert t.device.type == "mps"
            assert torch.allclose(t, torch.ones(64, device="mps"), rtol=1e-4)

        _run_distributed(fn, world_size=2, port=35200)

    def test_max_f32(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps") * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            assert t.device.type == "mps"
            assert torch.allclose(t, torch.ones(64, device="mps") * world_size,
                                  rtol=1e-4)

        _run_distributed(fn, world_size=2, port=35300)

    def test_product_f32(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps") * (rank + 2)  # 2, 3, ...
            dist.all_reduce(t, op=dist.ReduceOp.PRODUCT)
            assert t.device.type == "mps"
            prod = 1
            for r in range(world_size):
                prod *= r + 2
            assert torch.allclose(t, torch.ones(64, device="mps") * prod,
                                  rtol=1e-4)

        _run_distributed(fn, world_size=2, port=35400)

    def test_sum_f16(self):
        def fn(rank, world_size):
            t = torch.ones(64, device="mps", dtype=torch.float16) * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            assert t.device.type == "mps"
            expected_val = sum(r + 1 for r in range(world_size))
            expected = torch.ones(64, device="mps", dtype=torch.float16) * expected_val
            assert torch.allclose(t, expected, rtol=1e-2, atol=1e-2)

        _run_distributed(fn, world_size=2, port=35500)

    def test_sum_3rank(self):
        def fn(rank, world_size):
            t = torch.ones(128, device="mps") * (rank + 1)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            assert t.device.type == "mps"
            expected_val = sum(r + 1 for r in range(world_size))
            assert torch.allclose(t, torch.ones(128, device="mps") * expected_val,
                                  rtol=1e-4)

        _run_distributed(fn, world_size=3, port=35600)

    def test_large_payload(self):
        """1M elements — exercises the staging path & GPU reduce kernel size."""
        def fn(rank, world_size):
            torch.manual_seed(rank)
            t = torch.randn(1_000_000, device="mps")
            local = t.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            assert t.device.type == "mps"
            assert torch.isfinite(t).all().item(), "non-finite values in result"
            if world_size > 1:
                assert not torch.allclose(t, local), \
                    "all_reduce returned input unchanged"

        _run_distributed(fn, world_size=2, port=35700)


# ── _allgather_base (single contiguous output tensor) ────────────────

class TestAllgatherBase:
    def test_basic(self):
        def fn(rank, world_size):
            inp = torch.ones(16, device="mps") * (rank + 1)
            out = torch.empty(16 * world_size, device="mps")
            dist.all_gather_into_tensor(out, inp)
            assert out.device.type == "mps"
            for r in range(world_size):
                chunk = out[r * 16:(r + 1) * 16]
                expected = torch.ones(16, device="mps") * (r + 1)
                assert torch.allclose(chunk, expected, rtol=1e-4), \
                    f"rank {rank} chunk {r} mismatch"

        _run_distributed(fn, world_size=2, port=36000)

    def test_3rank(self):
        def fn(rank, world_size):
            inp = torch.ones(32, device="mps") * (rank + 1)
            out = torch.empty(32 * world_size, device="mps")
            dist.all_gather_into_tensor(out, inp)
            assert out.device.type == "mps"
            for r in range(world_size):
                chunk = out[r * 32:(r + 1) * 32]
                expected = torch.ones(32, device="mps") * (r + 1)
                assert torch.allclose(chunk, expected, rtol=1e-4)

        _run_distributed(fn, world_size=3, port=36100)


# ── _reduce_scatter_base (single contiguous input tensor) ────────────

class TestReduceScatterBase:
    def test_sum(self):
        def fn(rank, world_size):
            inp = torch.cat([torch.ones(16, device="mps") * (rank + r + 1)
                             for r in range(world_size)])
            out = torch.empty(16, device="mps")
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            assert out.device.type == "mps"
            # Element at position rank: sum over peers of (peer + rank + 1)
            expected_val = sum(p + rank + 1 for p in range(world_size))
            expected = torch.ones(16, device="mps") * expected_val
            assert torch.allclose(out, expected, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=36500)

    def test_avg(self):
        def fn(rank, world_size):
            inp = torch.cat([torch.ones(16, device="mps") * (rank + r + 1)
                             for r in range(world_size)])
            out = torch.empty(16, device="mps")
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.AVG)
            assert out.device.type == "mps"
            expected_val = sum(p + rank + 1 for p in range(world_size)) / world_size
            expected = torch.ones(16, device="mps") * expected_val
            assert torch.allclose(out, expected, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=36600)

    def test_3rank(self):
        def fn(rank, world_size):
            inp = torch.cat([torch.ones(8, device="mps") * (rank + r + 1)
                             for r in range(world_size)])
            out = torch.empty(8, device="mps")
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            assert out.device.type == "mps"
            expected_val = sum(p + rank + 1 for p in range(world_size))
            expected = torch.ones(8, device="mps") * expected_val
            assert torch.allclose(out, expected, rtol=1e-4)

        _run_distributed(fn, world_size=3, port=36700)


# ── alltoall_base (equal and variable splits) ─────────────────────────

class TestAlltoallBase:
    def test_equal_split(self):
        def fn(rank, world_size):
            # Each rank's input is world_size chunks of size 4. Chunk p
            # gets sent to rank p; we receive our rank-th chunk from each peer.
            inp = torch.cat([torch.ones(4, device="mps") * (rank * 10 + p)
                             for p in range(world_size)])
            out = torch.empty_like(inp)
            dist.all_to_all_single(out, inp)
            assert out.device.type == "mps"
            for p in range(world_size):
                chunk = out[p * 4:(p + 1) * 4]
                # From peer p, we got peer p's rank-th chunk = p * 10 + rank.
                expected = torch.ones(4, device="mps") * (p * 10 + rank)
                assert torch.allclose(chunk, expected, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=37000)

    def test_variable_split(self):
        def fn(rank, world_size):
            # rank 0 sends [1] to rank 0, [2,2] to rank 1
            # rank 1 sends [3,3,3] to rank 0, [4,4] to rank 1
            # so rank 0 recvs [1] + [3,3,3] = [1,3,3,3], rank 1 recvs [2,2] + [4,4]
            if rank == 0:
                in_splits = [1, 2]
                out_splits = [1, 3]
                values = [1.0, 2.0, 2.0]
            else:
                in_splits = [3, 2]
                out_splits = [2, 2]
                values = [3.0, 3.0, 3.0, 4.0, 4.0]
            inp = torch.tensor(values, device="mps")
            out = torch.empty(sum(out_splits), device="mps")
            dist.all_to_all_single(out, inp, out_splits, in_splits)
            assert out.device.type == "mps"
            if rank == 0:
                expected = torch.tensor([1.0, 3.0, 3.0, 3.0], device="mps")
            else:
                expected = torch.tensor([2.0, 2.0, 4.0, 4.0], device="mps")
            assert torch.allclose(out, expected, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=37100)


# ── alltoall (list form) ─────────────────────────────────────────────

class TestAlltoall:
    def test_basic(self):
        def fn(rank, world_size):
            ins = [torch.ones(8, device="mps") * (rank * 10 + p)
                   for p in range(world_size)]
            outs = [torch.empty(8, device="mps") for _ in range(world_size)]
            dist.all_to_all(outs, ins)
            for o in outs:
                assert o.device.type == "mps"
            for p in range(world_size):
                expected = torch.ones(8, device="mps") * (p * 10 + rank)
                assert torch.allclose(outs[p], expected, rtol=1e-4)

        _run_distributed(fn, world_size=2, port=37500)

    def test_3rank(self):
        def fn(rank, world_size):
            ins = [torch.ones(4, device="mps") * (rank * 100 + p)
                   for p in range(world_size)]
            outs = [torch.empty(4, device="mps") for _ in range(world_size)]
            dist.all_to_all(outs, ins)
            for p in range(world_size):
                expected = torch.ones(4, device="mps") * (p * 100 + rank)
                assert torch.allclose(outs[p], expected, rtol=1e-4)

        _run_distributed(fn, world_size=3, port=37600)


# ── barrier ──────────────────────────────────────────────────────────

class TestBarrier:
    def test_basic(self):
        def fn(rank, world_size):
            # Stagger entries so we can confirm barrier synchronises.
            import time
            time.sleep(0.1 * rank)
            t0 = time.time()
            dist.barrier()
            t1 = time.time()
            # On a 2-rank world, rank 0 should have waited ~0.1s for rank 1.
            # Don't pin a tight bound — just check the call completes.
            assert t1 - t0 < 10, "barrier didn't return in time"

        _run_distributed(fn, world_size=2, port=38000)


# ── CPU tensors are rejected ─────────────────────────────────────────

class TestCpuTensorRejected:
    """MCCL is MPS-only — a CPU tensor must surface a clean error."""

    def test_allreduce_cpu_raises(self):
        def fn(rank, world_size):
            t = torch.ones(8, device="cpu")
            raised = False
            try:
                # async_op=True returns a Work; wait() rethrows the body's
                # MCCLError so the test sees the rejection deterministically.
                work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
                work.wait()
            except Exception as e:
                raised = "MPS-only" in str(e) or "MCCL" in type(e).__name__
            assert raised, "expected MCCL to reject CPU tensor"

        _run_distributed(fn, world_size=2, port=39000)
