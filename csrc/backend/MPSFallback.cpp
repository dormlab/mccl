#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace mccl {

namespace {

void cpu_fallback_boxed(const c10::OperatorHandle& op,
                        c10::DispatchKeySet,
                        torch::jit::Stack* stack) {
    at::native::cpu_fallback(op, stack);
}

} // namespace

// Per-op MPS dispatch. Namespace-scoped `m.fallback(...)` was rejected
// on torch <= 2.7. Per-op `m.impl(...)` is the accepted form.
//
// Each op redispatches via at::native::cpu_fallback which moves any
// MPS tensor args to CPU, redispatches to the CPU key (which lands in
// our backend's allreduce/broadcast/etc.), then copies in-place outputs
// back to MPS. On Apple Silicon shared-storage tensors this is
// effectively zero-copy.
#define MCCL_IMPL_CPU_FB(name) \
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_boxed>())

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
    MCCL_IMPL_CPU_FB("allreduce_");
    MCCL_IMPL_CPU_FB("broadcast_");
    MCCL_IMPL_CPU_FB("_allgather_base_");
    MCCL_IMPL_CPU_FB("allgather_");
    MCCL_IMPL_CPU_FB("_reduce_scatter_base_");
    MCCL_IMPL_CPU_FB("reduce_scatter_");
    MCCL_IMPL_CPU_FB("alltoall_base_");
    MCCL_IMPL_CPU_FB("alltoall_");
    MCCL_IMPL_CPU_FB("send");
    MCCL_IMPL_CPU_FB("recv_");
    MCCL_IMPL_CPU_FB("barrier");
}

#undef MCCL_IMPL_CPU_FB

} // namespace mccl
