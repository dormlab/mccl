# Internal C++ tests

Component-level unit tests for the bits of the backend that don't go through torch — Progress engine, PeerMesh framing, wire utilities, Metal kernel correctness. **Not collective integration tests** — those live in `tests/collectives/` and exercise the c10d API from Python.

## Run

```bash
make -C tests/internal run
```

## Add a test

1. Drop `test_<thing>.cpp` next to the others.
2. `#include "_test.h"` and write `TEST_CASE("desc") { CHECK(cond); }` blocks. End the file with `MCCL_RUN_ALL()`.
3. Add the binary name to `TESTS = ...` in `Makefile` and list any `csrc/` source files the binary needs to link in (next to `Progress.cpp` in the existing recipe).

## What belongs here

| component | why C++ unit |
|---|---|
| `Progress` engine | thread + FIFO queue; doesn't need torch |
| `PeerMesh` framing helpers (e.g. score, subnet24) | pure utilities; testing from Python is silly |
| wire-protocol header parsing (`send`/`recv` 8-byte frame) | byte-level — Python wraps add nothing |
| Metal kernel correctness (`metal_reduce_op` on known inputs) | bypass `at::Tensor` glue; test kernels directly against fp32 buffers |

## What does *not* belong here

`dist.all_reduce` correctness, `init_process_group` behavior, mp.spawn flows — those test the user-facing API and stay in `tests/collectives/`.
