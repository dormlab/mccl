#pragma once

#include <torch/torch.h>
#include <c10d/Backend.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Work.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace distro {

class Rendezvous;

/// MCCL c10d backend for Apple Silicon clusters.
///
/// Construction does a Store-based rendezvous to exchange transport
/// endpoints; collectives are then dispatched via Metal (intra-node) and
/// the DMEM RDMA path (cross-node). For a single-rank world this becomes
/// a no-op wrapper that still produces real Work futures so the calling
/// code paths remain identical to multi-rank runs.
class ProcessGroupMCCL : public c10d::Backend {
public:
    struct Options {
        std::chrono::milliseconds timeout{30 * 60 * 1000};
        bool use_metal = true;   // use Metal kernels for local reduction
    };

    ProcessGroupMCCL(c10::intrusive_ptr<c10d::Store> store,
                     int rank,
                     int size,
                     Options opts = {});

    ~ProcessGroupMCCL() override;

    const std::string getBackendName() const override { return "mccl"; }

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& output_tensors,
        std::vector<at::Tensor>& input_tensors,
        const c10d::AllgatherOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> _allgather_base(
        at::Tensor& output_tensor,
        at::Tensor& input_tensor,
        const c10d::AllgatherOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor>& output_tensors,
        std::vector<std::vector<at::Tensor>>& input_tensors,
        const c10d::ReduceScatterOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> barrier(
        const c10d::BarrierOptions& opts = {}) override;

    c10::intrusive_ptr<c10d::Work> send(
        std::vector<at::Tensor>& tensors,
        int dst_rank,
        int tag) override;

    c10::intrusive_ptr<c10d::Work> recv(
        std::vector<at::Tensor>& tensors,
        int src_rank,
        int tag) override;

    /// Factory used by registration. Mirrors the `Backend::create_fn` signature
    /// expected by `torch.distributed.Backend.register_backend`.
    static c10::intrusive_ptr<c10d::Backend> create(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        const std::chrono::duration<float>& timeout);

private:
    c10::intrusive_ptr<c10d::Store> store_;
    Options opts_;
    std::unique_ptr<Rendezvous> rendezvous_;
    std::vector<std::string> endpoints_;
    bool metal_inited_ = false;

    /// Reduce `src` into `dst` in-place using Metal for MPS tensors and
    /// ATen ops for CPU tensors.
    void local_reduce_(at::Tensor& dst, const at::Tensor& src,
                       c10d::ReduceOp::RedOpType op);

    /// Apply scaling (used to convert SUM -> AVG after a SUM ring step).
    void local_scale_(at::Tensor& t, double scale);

    void ensure_metal_();
};

} // namespace distro
