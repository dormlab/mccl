import torch
import torch.distributed as dist


def test_import():
    import mccl  # noqa: F401


def test_backend_registered():
    import mccl  # noqa: F401
    assert "mccl" in dist.Backend.backend_type_map


def test_single_rank_no_op():
    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=0, world_size=1, store=dist.HashStore())
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        dist.all_reduce(x)
        assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0]))

        b = torch.tensor([7.0])
        dist.broadcast(b, src=0)
        assert torch.equal(b, torch.tensor([7.0]))

        dist.barrier()
    finally:
        dist.destroy_process_group()
