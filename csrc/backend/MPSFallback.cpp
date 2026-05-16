#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace distro {

namespace {

void cpu_fallback_boxed(const c10::OperatorHandle& op,
                        c10::DispatchKeySet,
                        torch::jit::Stack* stack) {
    at::native::cpu_fallback(op, stack);
}

} // namespace

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_boxed>());
}

} // namespace distro
