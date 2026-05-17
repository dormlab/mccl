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

// Namespace-scoped fallback is rejected on torch <= 2.7 with:
//   "Fallback functions which apply to only a single namespace
//    (you specified c10d) are not supported."
// Per-op TORCH_LIBRARY_IMPL("allreduce_", ...) etc. are required.
// Disabled here until those impls land. MPS tensors will continue
// to error at the c10d dispatcher boundary in the meantime.
// TORCH_LIBRARY_IMPL(c10d, MPS, m) {
//     m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_boxed>());
// }

} // namespace mccl
