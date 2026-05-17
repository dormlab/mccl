#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <cstddef>
#include <cstdint>

#include "metal/MPSInterop.hpp"

namespace mccl {

void metal_kernels_init();

/// dst += src element-wise on GPU via Metal compute.
void metal_accumulate_chunk(const at::Tensor& dst, const at::Tensor& src);

/// dst = min(dst, src) element-wise.
void metal_elementwise_min(const at::Tensor& dst, const at::Tensor& src);

/// dst = max(dst, src) element-wise.
void metal_elementwise_max(const at::Tensor& dst, const at::Tensor& src);

/// dst *= src element-wise.
void metal_elementwise_product(const at::Tensor& dst, const at::Tensor& src);

/// Dispatch the correct reduce kernel based on ReduceOp.
void metal_reduce_op(const at::Tensor& dst, const at::Tensor& src,
                     c10d::ReduceOp::RedOpType op);

/// Reuse a single command buffer across multiple GPU kernel launches.
/// Callers must end the batch before staging tensor data back to CPU/network.
void metal_begin_batch(const char* label = "mccl_batch");
void metal_end_batch();

/// buf *= scale element-wise on GPU via Metal compute.
void metal_scale_inplace(const at::Tensor& buf, double scale);

/// Fused: dst = (dst + src) * scale in a single kernel launch.
void metal_accumulate_and_scale(const at::Tensor& dst, const at::Tensor& src,
                                double scale);

/// Block until all MCCL Metal commands have completed.
void metal_sync();

/// View-based variants that take a pre-extracted MTLBuffer + offset + count
/// and a dtype. Use these from threads where touching torch's MPS allocator
/// state (`at::mps::getMTLBufferStorage`) would deadlock — e.g. the Progress
/// engine thread. Callers extract `MPSBufferView`s via `extract_mps_buffer`
/// once on the issuing thread and pass the views into the engine task.
void metal_reduce_op_view(const MPSBufferView& dst, const MPSBufferView& src,
                          at::ScalarType dtype, uint32_t count,
                          c10d::ReduceOp::RedOpType op);
void metal_scale_inplace_view(const MPSBufferView& buf, at::ScalarType dtype,
                              uint32_t count, double scale);

} // namespace mccl
