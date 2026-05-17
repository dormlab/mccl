# Tests

Three tiers, all under `pytest`. The shared `conftest.py` at the repo root provides two fixtures (`spawn`, `golden`) and editable constants (`WORLD_SIZES`, `FLOAT_DTYPES`, `INT_DTYPES`).

## Layout

```
tests/
├── unit/            # single-process, no networking
├── integration/     # mp.spawn N workers on loopback, one file per collective
├── multinode/       # cross-host SSH/torchrun launchers — manual
└── bench/           # perf size sweeps — opt-in via -m bench
```

## Running

```
mccl --test                              # unit + integration, default worlds 2,3
mccl --test --worlds 2,3,4,8             # widen the integration matrix
mccl test tests/integration -k allreduce # subset
mccl test -m slow                        # only slow tests

# pytest also works directly
pytest tests/unit
MCCL_TEST_WORLDS=2,4 pytest tests/integration
```

World sizes are runtime-configurable via `MCCL_TEST_WORLDS` (comma-separated) or `--worlds`. Same harness scales to any N — `mp.spawn` fans out workers on one host.

## Adding a new collective test

Drop a new file in `tests/integration/`:

```python
import pytest
import torch
import torch.distributed as dist
from conftest import WORLD_SIZES, ALL_DTYPES

def _worker(rank, world, dtype):
    x = ...   # set up input
    dist.your_new_op(x)
    return x.cpu()

@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_your_new_op(spawn, golden, world_size, dtype):
    res = spawn(world_size, _worker, dtype=dtype)
    expected = golden.your_new_op(...)
    for rank, got in res.items():
        assert torch.allclose(got.float(), expected)
```

That's it. If a new golden is needed, add it in `conftest.py` next to the others.

## Extending coverage

- More world sizes: append to `WORLD_SIZES` in `conftest.py`. Every parametrized test picks them up.
- More dtypes: append to `FLOAT_DTYPES` / `INT_DTYPES`.
- More reduce ops: append to `REDUCE_OPS_ALL`.

## Multinode

`tests/multinode/bench.py` is the existing cross-machine runner. Launch one process per rank with `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` env vars set. Example for 3 minis (run on each):

```
MASTER_ADDR=<rank0-tb-ip> MASTER_PORT=29500 RANK=$N WORLD_SIZE=3 \
  python tests/multinode/bench.py --rank $N --world 3 --master <rank0-tb-ip>
```
