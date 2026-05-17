"""Single-process sanity checks — no networking."""
import os

import torch
import torch.distributed as dist


def test_import():
    import mccl  # noqa: F401


def test_backend_registered():
    import mccl  # noqa: F401
    assert "mccl" in dist.Backend.backend_type_map


def test_default_iface_priority_set():
    import mccl  # noqa: F401
    assert os.environ.get("MCCL_IFACE_PRIORITY"), \
        "default MCCL_IFACE_PRIORITY should be set after import mccl"


def test_set_iface_priority():
    import mccl
    mccl.set_iface_priority(["192.168.103.", "192.168.102."])
    assert os.environ["MCCL_IFACE_PRIORITY"] == "192.168.103.,192.168.102."


def test_single_rank_no_op():
    """world=1 fast path: every collective should be a clean no-op."""
    import mccl  # noqa: F401
    dist.init_process_group(backend="mccl", rank=0, world_size=1,
                            store=dist.HashStore())
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        dist.all_reduce(x); assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0]))
        b = torch.tensor([7.0])
        dist.broadcast(b, src=0); assert torch.equal(b, torch.tensor([7.0]))
        dist.barrier()
    finally:
        dist.destroy_process_group()
