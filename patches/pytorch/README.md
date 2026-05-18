# PyTorch patches required by mccl

## 0001-MPSStream-flush-defensive-status-check.patch

**File:** `aten/src/ATen/mps/MPSStream.mm`

**What it fixes.** `MPSStream::flush()` unconditionally calls `[_commandBuffer commit]`. When an external collective backend (mccl) records events on the current MPS stream during DDP backward, MPSGraph's async encoding can leave the underlying `MTLCommandBuffer` in a committed state by the time `flush()` runs. The unguarded commit then trips Metal's `"commit an already committed command buffer"` assertion and the process aborts.

The patch wraps the commit in a status check — only commit if the buffer is in `NotEnqueued` state. Twenty lines of context, two lines of real change.

## How to apply

```bash
git clone --depth 1 --branch v2.12.0 --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/pytorch.git
cd pytorch
patch -p0 < /path/to/mccl/patches/pytorch/0001-MPSStream-flush-defensive-status-check.patch
brew install cmake ninja
MAX_JOBS=4 USE_DISTRIBUTED=1 USE_MPS=1 BUILD_TEST=0 USE_CUDA=0 USE_ROCM=0 \
  python setup.py bdist_wheel
pip install --force-reinstall --no-deps dist/torch-*.whl
```

Build takes ~45–90 min on an M4 16GB. Same wheel works on all minis (same CPU + macOS).

## Why we don't ship a fork

The patch is tiny and surgical. Maintaining a fork of pytorch is a lot of overhead for two lines of code. Instead we keep the patch in this repo and expect users on the mccl-3-mini setup to apply it once when they install. If/when this lands upstream we'll drop the patch.

## Without the patch

mccl still works — `ProcessGroupMCCL::allreduce` runs inline-sync (no cross-stream event recording, no race). You lose the ability to use the async overlap path. With the patch the async wiring is safe to enable, but in practice we ship inline-sync anyway because the variance is too high (see `csrc/backend/ProcessGroupMCCL.cpp::allreduce` comment).
