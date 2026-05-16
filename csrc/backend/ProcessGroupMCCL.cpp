#include "backend/ProcessGroupMCCL.hpp"
#include "backend/WorkMCCL.hpp"
#include "runtime/Rendezvous.hpp"
#include "common/Errors.hpp"

#include <ATen/ATen.h>

#ifdef DISTRO_HAS_METAL_KERNELS
#include "metal/MetalKernels.hpp"
#endif

namespace distro {

namespace {

c10::intrusive_ptr<c10d::Work> make_completed(
    c10d::OpType op, std::vector<at::Tensor> outputs, const char* title) {
    auto w = c10::make_intrusive<WorkMCCL>(op, std::move(outputs), title);
    w->finish();
    return w;
}

c10::intrusive_ptr<c10d::Work> make_failed(
    c10d::OpType op, std::vector<at::Tensor> outputs,
    const char* title, const std::string& msg) {
    auto w = c10::make_intrusive<WorkMCCL>(op, std::move(outputs), title);
    try {
        throw MCCLError(msg);
    } catch (...) {
        w->finishWithException(std::current_exception());
    }
    return w;
}

} // namespace

ProcessGroupMCCL::ProcessGroupMCCL(c10::intrusive_ptr<c10d::Store> store,
                                   int rank, int size, Options opts)
    : c10d::Backend(rank, size),
      store_(std::move(store)),
      opts_(opts) {
    DISTRO_CHECK(rank >= 0 && rank < size, "rank/size invalid");
    if (size > 1) {
        rendezvous_ = std::make_unique<Rendezvous>(
            store_, rank, size, opts_.timeout);
        // Endpoint exchange is a no-op placeholder until the DMEM transport
        // is plumbed through; the rendezvous is constructed here so the
        // Store handle's lifetime is tied to the PG.
    }
}

ProcessGroupMCCL::~ProcessGroupMCCL() = default;

void ProcessGroupMCCL::ensure_metal_() {
#ifdef DISTRO_HAS_METAL_KERNELS
    if (!metal_inited_ && opts_.use_metal) {
        metal_kernels_init();
        metal_inited_ = true;
    }
#endif
}

void ProcessGroupMCCL::local_reduce_(at::Tensor& dst, const at::Tensor& src,
                                     c10d::ReduceOp::RedOpType op) {
#ifdef DISTRO_HAS_METAL_KERNELS
    if (dst.device().is_mps() && src.device().is_mps() && opts_.use_metal) {
        ensure_metal_();
        metal_reduce_op(dst, src, op);
        return;
    }
#endif
    switch (op) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
            dst.add_(src);
            break;
        case c10d::ReduceOp::PRODUCT:
            dst.mul_(src);
            break;
        case c10d::ReduceOp::MIN:
            at::min_out(dst, dst, src);
            break;
        case c10d::ReduceOp::MAX:
            at::max_out(dst, dst, src);
            break;
        default:
            throw MCCLError("unsupported ReduceOp");
    }
}

void ProcessGroupMCCL::local_scale_(at::Tensor& t, double scale) {
    if (scale == 1.0) return;
#ifdef DISTRO_HAS_METAL_KERNELS
    if (t.device().is_mps() && opts_.use_metal) {
        ensure_metal_();
        metal_scale_inplace(t, scale);
        return;
    }
#endif
    t.mul_(scale);
}

// ─── Collectives ──────────────────────────────────────────────────────────
//
// Single-rank world is a no-op fast path: input is already the result.
// Multi-rank execution is gated behind the DMEM transport, which is wired
// up in a follow-up change; for now we fail loudly with a clear message
// instead of silently producing wrong results.

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& /*opts*/) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce");
    }
    return make_failed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce",
        "multi-rank allreduce not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& /*opts*/) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::BROADCAST, tensors, "mccl:broadcast");
    }
    return make_failed(c10d::OpType::BROADCAST, tensors, "mccl:broadcast",
        "multi-rank broadcast not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& /*opts*/) {
    if (getSize() == 1) {
        DISTRO_CHECK(!output_tensors.empty() && !output_tensors[0].empty(),
                     "allgather output empty");
        output_tensors[0][0].copy_(input_tensors[0]);
        return make_completed(c10d::OpType::ALLGATHER,
                              output_tensors[0], "mccl:allgather");
    }
    return make_failed(c10d::OpType::ALLGATHER, input_tensors, "mccl:allgather",
        "multi-rank allgather not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& /*opts*/) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_ALLGATHER_BASE,
                              {output_tensor}, "mccl:_allgather_base");
    }
    return make_failed(c10d::OpType::_ALLGATHER_BASE, {output_tensor},
        "mccl:_allgather_base",
        "multi-rank _allgather_base not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& /*opts*/) {
    if (getSize() == 1) {
        DISTRO_CHECK(!input_tensors.empty() && !input_tensors[0].empty(),
                     "reduce_scatter input empty");
        output_tensors[0].copy_(input_tensors[0][0]);
        return make_completed(c10d::OpType::REDUCE_SCATTER,
                              output_tensors, "mccl:reduce_scatter");
    }
    return make_failed(c10d::OpType::REDUCE_SCATTER, output_tensors,
        "mccl:reduce_scatter",
        "multi-rank reduce_scatter not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::barrier(
    const c10d::BarrierOptions& /*opts*/) {
    if (getSize() > 1 && rendezvous_) {
        try {
            rendezvous_->barrier("pg_barrier");
        } catch (...) {
            auto w = c10::make_intrusive<WorkMCCL>(
                c10d::OpType::BARRIER, std::vector<at::Tensor>{}, "mccl:barrier");
            w->finishWithException(std::current_exception());
            return w;
        }
    }
    return make_completed(c10d::OpType::BARRIER, {}, "mccl:barrier");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors, int /*dst_rank*/, int /*tag*/) {
    return make_failed(c10d::OpType::SEND, tensors, "mccl:send",
        "point-to-point send not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors, int /*src_rank*/, int /*tag*/) {
    return make_failed(c10d::OpType::RECV, tensors, "mccl:recv",
        "point-to-point recv not yet wired to DMEM transport");
}

c10::intrusive_ptr<c10d::Backend> ProcessGroupMCCL::create(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank, int size,
    const std::chrono::duration<float>& timeout) {
    Options opts;
    opts.timeout = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
    return c10::make_intrusive<ProcessGroupMCCL>(store, rank, size, opts);
}

} // namespace distro
