# Tests

## Layout

```
tests/
├── common.py             # shared harness — spawn, golden refs, DTYPES, REDUCE_OPS
├── conftest.py           # pytest fixtures exposing common.py helpers
├── test_smoke.py         # single-process sanity (import, register, single-rank)
├── collectives/          # one file per c10d op
│   ├── test_allreduce.py
│   ├── test_broadcast.py
│   ├── test_allgather.py
│   ├── test_reduce_scatter.py
│   ├── test_alltoall.py
│   ├── test_p2p.py
│   └── test_barrier.py
└── multinode/            # cross-host launchers (manual)
```

Single source of truth lives in `common.py`:

- `WORLD` — int read from `MCCL_TEST_WORLD` (default `2`).
- `DTYPES` — every torch+MPS dtype the backend can move; reduce ops skip non-float.
- `REDUCE_OPS` — `SUM, MIN, MAX, PRODUCT, AVG`.
- `spawn(fn, **kw)` — fan out workers via `mp.spawn`, surface exceptions as pytest failures.
- `seed_tensor` / `golden_*` — deterministic inputs and pure-Python expected results.
- `assert_close` — dtype-aware tolerance.

## Run

```
./setup                                      # one-time per machine
pytest tests/test_smoke.py                   # ~3s, no networking
MCCL_TEST_WORLD=2 pytest tests/collectives/  # default, fastest
MCCL_TEST_WORLD=4 pytest tests/collectives/  # raise N at runtime; any N ≥ 2
pytest tests/collectives/ -k allreduce       # subset
```

## Adding things

- **New collective:** new file in `tests/collectives/` shaped like the others — module-level worker, `parametrize(dtype, …)`, call `spawn`, compare to a golden.
- **New dtype:** append to `DTYPES` in `common.py`. Every test picks it up. Non-float dtypes auto-skip in reduce tests via `is_reduce_dtype`.
- **In-place / out-of-place:** already parametrized on `in_place` for collectives where it applies (allgather, reduce_scatter, alltoall_base).

## Multi-host

`tests/multinode/bench.py` is the cross-machine runner. Launch one process per rank with `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` set:

```
RANK=$N WORLD_SIZE=3 MASTER_ADDR=<rank0-tb-ip> MASTER_PORT=29500 \
  python tests/multinode/bench.py --rank $N --world 3 --master <rank0-tb-ip>
```
