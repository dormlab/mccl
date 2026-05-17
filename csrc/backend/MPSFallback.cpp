#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace mccl {

namespace {

// Re-resolve dispatch with the CPU key added to the keyset. The dispatcher
// then picks c10d's CPU-key generic boxed shim — which pulls the
// ProcessGroup off the op stack and calls the matching PG virtual — but
// the tensors in the stack stay on MPS. The PG body handles them through
// Metal kernels (see ProcessGroupMCCL.cpp); no host bounce.
void mps_native_redispatch_boxed(const c10::OperatorHandle& op,
                                 c10::DispatchKeySet ks,
                                 torch::jit::Stack* stack) {
    op.redispatchBoxed(ks | c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
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
