#include "backend/ProcessGroupMCCL.hpp"
#include "backend/WorkMCCL.hpp"
#include "backend/PeerMesh.hpp"
#include "runtime/Rendezvous.hpp"
#include "common/Errors.hpp"

#include <ATen/ATen.h>

#include <cstring>
#include <string>
#include <thread>

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
    const char* title, std::exception_ptr eptr) {
    auto w = c10::make_intrusive<WorkMCCL>(op, std::move(outputs), title);
    w->finishWithException(eptr);
    return w;
}

c10::intrusive_ptr<c10d::Work> make_unimplemented(
    c10d::OpType op, std::vector<at::Tensor> outputs, const char* title) {
    return make_failed(op, std::move(outputs), title,
        std::make_exception_ptr(MCCLError(
            std::string(title) + ": not implemented")));
}

std::vector<uint8_t> tensor_to_bytes(const at::Tensor& t) {
    auto c = t.contiguous().cpu();
    std::vector<uint8_t> b(c.nbytes());
    std::memcpy(b.data(), c.data_ptr(), b.size());
    return b;
}

at::Tensor bytes_to_cpu_like(const std::vector<uint8_t>& b, const at::Tensor& tmpl) {
    auto t = at::empty(tmpl.sizes(), tmpl.options().device(at::kCPU));
    DISTRO_CHECK(b.size() == static_cast<size_t>(t.nbytes()), "size mismatch");
    std::memcpy(t.data_ptr(), b.data(), b.size());
    return t;
}

void cpu_reduce_(at::Tensor& dst, const at::Tensor& src, c10d::ReduceOp::RedOpType op) {
    switch (op) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:     dst.add_(src); break;
        case c10d::ReduceOp::PRODUCT: dst.mul_(src); break;
        case c10d::ReduceOp::MIN:     at::min_out(dst, dst, src); break;
        case c10d::ReduceOp::MAX:     at::max_out(dst, dst, src); break;
        default: throw MCCLError("unsupported ReduceOp");
    }
}

std::string key(const char* op, uint64_t seq, int r) {
    return std::string("mccl/") + op + "/" + std::to_string(seq) + "/" + std::to_string(r);
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
        mesh_ = std::make_unique<PeerMesh>(
            store_, rank, size, opts_.timeout);
    }
}

ProcessGroupMCCL::~ProcessGroupMCCL() = default;

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce");
    }
    try {
        auto op = opts.reduceOp;
        for (auto& t : tensors) {
            auto src = t.contiguous().cpu();
            auto acc = src.is_same(t) ? src : src.clone();
            size_t nbytes = acc.nbytes();
            std::vector<at::Tensor> recvbufs(getSize());
            std::vector<std::thread> ts;
            ts.reserve(getSize() - 1);
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                recvbufs[p] = at::empty_like(acc);
                ts.emplace_back([&, p] {
                    mesh_->send(p, acc.data_ptr(), nbytes);
                    mesh_->recv(p, recvbufs[p].data_ptr(), nbytes);
                });
            }
            for (auto& th : ts) th.join();
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                cpu_reduce_(acc, recvbufs[p], op);
            }
            if (op == c10d::ReduceOp::AVG) acc.div_(getSize());
            if (!acc.is_same(t)) t.copy_(acc);
        }
    } catch (...) {
        return make_failed(c10d::OpType::ALLREDUCE, tensors,
                           "mccl:allreduce", std::current_exception());
    }
    return make_completed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::BROADCAST, tensors, "mccl:broadcast");
    }
    try {
        int root = opts.rootRank;
        for (auto& t : tensors) {
            auto cpu = t.contiguous().cpu();
            size_t nbytes = cpu.nbytes();
            if (getRank() == root) {
                std::vector<std::thread> ts;
                for (int p = 0; p < getSize(); ++p) {
                    if (p == root) continue;
                    ts.emplace_back([&, p] {
                        mesh_->send(p, cpu.data_ptr(), nbytes);
                    });
                }
                for (auto& th : ts) th.join();
            } else {
                mesh_->recv(root, cpu.data_ptr(), nbytes);
                t.copy_(cpu);
            }
        }
    } catch (...) {
        return make_failed(c10d::OpType::BROADCAST, tensors,
                           "mccl:broadcast", std::current_exception());
    }
    return make_completed(c10d::OpType::BROADCAST, tensors, "mccl:broadcast");
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
    try {
        DISTRO_CHECK(input_tensors.size() == output_tensors.size(),
                     "allgather input/output count mismatch");
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            uint64_t s = next_seq_();
            auto& outs = output_tensors[i];
            DISTRO_CHECK(static_cast<int>(outs.size()) == getSize(),
                         "allgather output list size != world");
            store_->set(key("ag", s, getRank()), tensor_to_bytes(input_tensors[i]));
            for (int p = 0; p < getSize(); ++p) {
                auto k = key("ag", s, p);
                if (p != getRank()) store_->wait({k}, opts_.timeout);
                outs[p].copy_(bytes_to_cpu_like(store_->get(k), outs[p]));
            }
        }
    } catch (...) {
        return make_failed(c10d::OpType::ALLGATHER, input_tensors,
                           "mccl:allgather", std::current_exception());
    }
    return make_completed(c10d::OpType::ALLGATHER, input_tensors, "mccl:allgather");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& /*opts*/) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_ALLGATHER_BASE,
                              {output_tensor}, "mccl:_allgather_base");
    }
    try {
        uint64_t s = next_seq_();
        store_->set(key("agb", s, getRank()), tensor_to_bytes(input_tensor));
        auto chunk_bytes = input_tensor.nbytes();
        auto* dst = static_cast<uint8_t*>(output_tensor.contiguous().data_ptr());
        auto cpu_out = at::empty(output_tensor.sizes(),
                                 output_tensor.options().device(at::kCPU));
        auto* cpu_dst = static_cast<uint8_t*>(cpu_out.data_ptr());
        for (int p = 0; p < getSize(); ++p) {
            auto k = key("agb", s, p);
            if (p != getRank()) store_->wait({k}, opts_.timeout);
            auto b = store_->get(k);
            DISTRO_CHECK(b.size() == chunk_bytes, "_allgather_base chunk size mismatch");
            std::memcpy(cpu_dst + p * chunk_bytes, b.data(), chunk_bytes);
        }
        output_tensor.copy_(cpu_out);
        (void)dst;
    } catch (...) {
        return make_failed(c10d::OpType::_ALLGATHER_BASE, {output_tensor},
                           "mccl:_allgather_base", std::current_exception());
    }
    return make_completed(c10d::OpType::_ALLGATHER_BASE,
                          {output_tensor}, "mccl:_allgather_base");
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
    return make_unimplemented(c10d::OpType::REDUCE_SCATTER, output_tensors,
                              "mccl:reduce_scatter");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::barrier(
    const c10d::BarrierOptions& /*opts*/) {
    if (getSize() > 1 && rendezvous_) {
        try {
            rendezvous_->barrier("pg_barrier_" + std::to_string(next_seq_()));
        } catch (...) {
            return make_failed(c10d::OpType::BARRIER, {},
                               "mccl:barrier", std::current_exception());
        }
    }
    return make_completed(c10d::OpType::BARRIER, {}, "mccl:barrier");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors, int /*dst_rank*/, int /*tag*/) {
    return make_unimplemented(c10d::OpType::SEND, tensors, "mccl:send");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors, int /*src_rank*/, int /*tag*/) {
    return make_unimplemented(c10d::OpType::RECV, tensors, "mccl:recv");
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
