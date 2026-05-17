#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace mccl {

namespace {

// Re-resolve the c10d op on the CPU dispatch key. c10d ships a generic
// CPU-key boxed shim for every collective that pulls the ProcessGroup
// off the op stack and calls the matching PG virtual. By redispatching
// onto CPU we land in that shim with the tensors still on MPS — the PG
// body (see ProcessGroupMCCL.cpp) handles them through Metal kernels
// and zero-copy MTLBuffer staging.
//
// The keyset passed must contain ONLY CPU. Anything else (including any
// MPS-flavoured key already in the original `ks`) makes the dispatcher
// pick our own MPS impl again, infinite-looping.
void mps_native_redispatch_boxed(const c10::OperatorHandle& op,
                                 c10::DispatchKeySet,
                                 torch::jit::Stack* stack) {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
}

} // namespace

#define MCCL_IMPL_MPS_NATIVE(name) \
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&mps_native_redispatch_boxed>())

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
    MCCL_IMPL_MPS_NATIVE("allreduce_");
    MCCL_IMPL_MPS_NATIVE("broadcast_");
    MCCL_IMPL_MPS_NATIVE("_allgather_base_");
    MCCL_IMPL_MPS_NATIVE("allgather_");
    MCCL_IMPL_MPS_NATIVE("_reduce_scatter_base_");
    MCCL_IMPL_MPS_NATIVE("reduce_scatter_");
    MCCL_IMPL_MPS_NATIVE("alltoall_base_");
    MCCL_IMPL_MPS_NATIVE("alltoall_");
    MCCL_IMPL_MPS_NATIVE("send");
    MCCL_IMPL_MPS_NATIVE("recv_");
    MCCL_IMPL_MPS_NATIVE("barrier");
}

#undef MCCL_IMPL_MPS_NATIVE

} // namespace mccl
