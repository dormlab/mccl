import time
import torch.distributed as dist


def _worker(rank, world):
    time.sleep(0.05 * rank)
    t0 = time.perf_counter()
    dist.barrier()
    return time.perf_counter() - t0


def test_barrier(spawn):
    res = spawn(_worker)
    for _, dt in res.items():
        assert dt < 10.0, f"barrier returned in {dt:.2f}s"
