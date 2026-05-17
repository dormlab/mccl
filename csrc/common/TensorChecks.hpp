#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include "common/Errors.hpp"

namespace mccl {

inline void check_single_tensor(const at::Tensor& tensor) {
    DISTRO_CHECK_TENSOR(
        tensor.is_mps() || tensor.is_cpu(),
        "MCCL requires MPS or CPU tensors (unified memory), got device: " + tensor.device().str()
    );

    DISTRO_CHECK_TENSOR(
        tensor.is_contiguous(),
        "MCCL v1 requires contiguous tensors. "
        "Call .contiguous() before passing to collective."
    );

    auto dtype = tensor.scalar_type();
    DISTRO_CHECK_TENSOR(
        dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 ||
        dtype == at::kDouble ||
        dtype == at::kLong || dtype == at::kInt || dtype == at::kShort ||
        dtype == at::kChar || dtype == at::kByte || dtype == at::kBool,
        "MCCL: unsupported dtype " + std::string(at::toString(dtype))
    );

    DISTRO_CHECK_TENSOR(
        tensor.numel() > 0,
        "MCCL does not accept empty tensors"
    );
}

inline void check_tensor_list(const std::vector<at::Tensor>& tensors) {
    DISTRO_CHECK_TENSOR(
        tensors.size() == 1,
        "MCCL collectives expect exactly one tensor per rank per call"
    );
    check_single_tensor(tensors[0]);
}

inline void check_same_shape_dtype(const at::Tensor& a, const at::Tensor& b) {
    DISTRO_CHECK_TENSOR(
        a.sizes() == b.sizes(),
        "Shape mismatch in collective"
    );
    DISTRO_CHECK_TENSOR(
        a.scalar_type() == b.scalar_type(),
        "Dtype mismatch in collective"
    );
}

inline size_t tensor_nbytes(const at::Tensor& tensor) {
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

} // namespace mccl
