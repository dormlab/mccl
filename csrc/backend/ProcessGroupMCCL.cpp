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

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& /*opts*/) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_ALLGATHER_BASE,
                              {output_tensor}, "mccl:_allgather_base");
    }
    try {
        auto in_cpu = input_tensor.contiguous().cpu();
        size_t chunk = in_cpu.nbytes();
        DISTRO_CHECK(output_tensor.nbytes() == chunk * getSize(),
                     "_allgather_base: output size != input * world");
        auto out_cpu = at::empty(output_tensor.sizes(),
                                 output_tensor.options().device(at::kCPU));
        auto* dst = static_cast<uint8_t*>(out_cpu.data_ptr());
        std::memcpy(dst + getRank() * chunk, in_cpu.data_ptr(), chunk);
        std::vector<std::thread> ts;
        ts.reserve(getSize() - 1);
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            ts.emplace_back([&, p] {
                mesh_->send(p, in_cpu.data_ptr(), chunk);
                mesh_->recv(p, dst + p * chunk, chunk);
            });
        }
        for (auto& th : ts) th.join();
        output_tensor.copy_(out_cpu);
    } catch (...) {
        return make_failed(c10d::OpType::_ALLGATHER_BASE, {output_tensor},
                           "mccl:_allgather_base", std::current_exception());
    }
    return make_completed(c10d::OpType::_ALLGATHER_BASE,
                          {output_tensor}, "mccl:_allgather_base");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {
    if (getSize() == 1) {
        DISTRO_CHECK(!output_tensors.empty() && !output_tensors[0].empty(),
                     "allgather output empty");
        output_tensors[0][0].copy_(input_tensors[0]);
        return make_completed(c10d::OpType::ALLGATHER,
                              output_tensors[0], "mccl:allgather");
    }
    try {
        DISTRO_CHECK(input_tensors.size() == output_tensors.size(),
                     "allgather: input/output count mismatch");
        for (size_t i = 0; i < input_tensors.size(); ++i) {
            auto& in = input_tensors[i];
            auto& outs = output_tensors[i];
            DISTRO_CHECK(static_cast<int>(outs.size()) == getSize(),
                         "allgather: output list size != world");
            auto in_cpu = in.contiguous().cpu();
            size_t chunk = in_cpu.nbytes();
            std::vector<at::Tensor> peer_cpu(getSize());
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                peer_cpu[p] = at::empty(outs[p].sizes(),
                                        outs[p].options().device(at::kCPU));
            }
            std::vector<std::thread> ts;
            ts.reserve(getSize() - 1);
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                ts.emplace_back([&, p] {
                    mesh_->send(p, in_cpu.data_ptr(), chunk);
                    mesh_->recv(p, peer_cpu[p].data_ptr(), chunk);
                });
            }
            for (auto& th : ts) th.join();
            outs[getRank()].copy_(in_cpu);
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                outs[p].copy_(peer_cpu[p]);
            }
        }
        (void)opts;
    } catch (...) {
        return make_failed(c10d::OpType::ALLGATHER, input_tensors,
                           "mccl:allgather", std::current_exception());
    }
    return make_completed(c10d::OpType::ALLGATHER, input_tensors, "mccl:allgather");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_reduce_scatter_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::ReduceScatterOptions& opts) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_REDUCE_SCATTER_BASE,
                              {output_tensor}, "mccl:_reduce_scatter_base");
    }
    try {
        auto op = opts.reduceOp;
        auto in_cpu = input_tensor.contiguous().cpu();
        size_t chunk = output_tensor.nbytes();
        DISTRO_CHECK(in_cpu.nbytes() == chunk * getSize(),
                     "_reduce_scatter_base: input size != output * world");
        auto* src = static_cast<uint8_t*>(in_cpu.data_ptr());
        auto acc = at::empty(output_tensor.sizes(),
                             output_tensor.options().device(at::kCPU));
        std::memcpy(acc.data_ptr(), src + getRank() * chunk, chunk);
        std::vector<at::Tensor> recvbufs(getSize());
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            recvbufs[p] = at::empty(output_tensor.sizes(),
                                    output_tensor.options().device(at::kCPU));
        }
        std::vector<std::thread> ts;
        ts.reserve(getSize() - 1);
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            ts.emplace_back([&, p] {
                mesh_->send(p, src + p * chunk, chunk);
                mesh_->recv(p, recvbufs[p].data_ptr(), chunk);
            });
        }
        for (auto& th : ts) th.join();
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            cpu_reduce_(acc, recvbufs[p], op);
        }
        if (op == c10d::ReduceOp::AVG) acc.div_(getSize());
        output_tensor.copy_(acc);
    } catch (...) {
        return make_failed(c10d::OpType::_REDUCE_SCATTER_BASE, {output_tensor},
                           "mccl:_reduce_scatter_base", std::current_exception());
    }
    return make_completed(c10d::OpType::_REDUCE_SCATTER_BASE,
                          {output_tensor}, "mccl:_reduce_scatter_base");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
    if (getSize() == 1) {
        DISTRO_CHECK(!input_tensors.empty() && !input_tensors[0].empty(),
                     "reduce_scatter input empty");
        output_tensors[0].copy_(input_tensors[0][0]);
        return make_completed(c10d::OpType::REDUCE_SCATTER,
                              output_tensors, "mccl:reduce_scatter");
    }
    try {
        auto op = opts.reduceOp;
        DISTRO_CHECK(output_tensors.size() == input_tensors.size(),
                     "reduce_scatter: count mismatch");
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& outs = output_tensors[i];
            auto& ins = input_tensors[i];
            DISTRO_CHECK(static_cast<int>(ins.size()) == getSize(),
                         "reduce_scatter: input list size != world");
            std::vector<at::Tensor> ins_cpu(getSize());
            for (int p = 0; p < getSize(); ++p) ins_cpu[p] = ins[p].contiguous().cpu();
            size_t chunk = ins_cpu[0].nbytes();
            auto acc = ins_cpu[getRank()].clone();
            std::vector<at::Tensor> recvbufs(getSize());
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                recvbufs[p] = at::empty_like(ins_cpu[getRank()]);
            }
            std::vector<std::thread> ts;
            ts.reserve(getSize() - 1);
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                ts.emplace_back([&, p] {
                    mesh_->send(p, ins_cpu[p].data_ptr(), chunk);
                    mesh_->recv(p, recvbufs[p].data_ptr(), chunk);
                });
            }
            for (auto& th : ts) th.join();
            for (int p = 0; p < getSize(); ++p) {
                if (p == getRank()) continue;
                cpu_reduce_(acc, recvbufs[p], op);
            }
            if (op == c10d::ReduceOp::AVG) acc.div_(getSize());
            outs.copy_(acc);
        }
    } catch (...) {
        return make_failed(c10d::OpType::REDUCE_SCATTER, output_tensors,
                           "mccl:reduce_scatter", std::current_exception());
    }
    return make_completed(c10d::OpType::REDUCE_SCATTER, output_tensors,
                          "mccl:reduce_scatter");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::alltoall_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10d::AllToAllOptions& /*opts*/) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::ALLTOALL_BASE,
                              {output_tensor}, "mccl:alltoall_base");
    }
    try {
        auto in_cpu = input_tensor.contiguous().cpu();
        auto out_cpu = at::empty(output_tensor.sizes(),
                                 output_tensor.options().device(at::kCPU));
        size_t esize = static_cast<size_t>(in_cpu.element_size());
        DISTRO_CHECK(esize == static_cast<size_t>(out_cpu.element_size()),
                     "alltoall_base: dtype mismatch");

        std::vector<size_t> in_off(getSize() + 1, 0), out_off(getSize() + 1, 0);
        if (input_split_sizes.empty()) {
            size_t per = in_cpu.numel() / getSize() * esize;
            DISTRO_CHECK(per * getSize() == in_cpu.nbytes(),
                         "alltoall_base: input not divisible by world");
            for (int p = 0; p < getSize(); ++p) in_off[p + 1] = in_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(input_split_sizes.size()) == getSize(),
                         "alltoall_base: input_split_sizes wrong length");
            for (int p = 0; p < getSize(); ++p)
                in_off[p + 1] = in_off[p] + input_split_sizes[p] * esize;
        }
        if (output_split_sizes.empty()) {
            size_t per = out_cpu.numel() / getSize() * esize;
            DISTRO_CHECK(per * getSize() == out_cpu.nbytes(),
                         "alltoall_base: output not divisible by world");
            for (int p = 0; p < getSize(); ++p) out_off[p + 1] = out_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(output_split_sizes.size()) == getSize(),
                         "alltoall_base: output_split_sizes wrong length");
            for (int p = 0; p < getSize(); ++p)
                out_off[p + 1] = out_off[p] + output_split_sizes[p] * esize;
        }

        auto* src = static_cast<uint8_t*>(in_cpu.data_ptr());
        auto* dst = static_cast<uint8_t*>(out_cpu.data_ptr());

        size_t self_in_len  = in_off[getRank() + 1]  - in_off[getRank()];
        size_t self_out_len = out_off[getRank() + 1] - out_off[getRank()];
        DISTRO_CHECK(self_in_len == self_out_len,
                     "alltoall_base: self chunk size mismatch");
        std::memcpy(dst + out_off[getRank()], src + in_off[getRank()], self_in_len);

        std::vector<std::thread> ts;
        ts.reserve(getSize() - 1);
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            size_t slen = in_off[p + 1]  - in_off[p];
            size_t rlen = out_off[p + 1] - out_off[p];
            if (slen == 0 && rlen == 0) continue;
            ts.emplace_back([&, p, slen, rlen] {
                if (slen) mesh_->send(p, src + in_off[p], slen);
                if (rlen) mesh_->recv(p, dst + out_off[p], rlen);
            });
        }
        for (auto& th : ts) th.join();
        output_tensor.copy_(out_cpu);
    } catch (...) {
        return make_failed(c10d::OpType::ALLTOALL_BASE, {output_tensor},
                           "mccl:alltoall_base", std::current_exception());
    }
    return make_completed(c10d::OpType::ALLTOALL_BASE,
                          {output_tensor}, "mccl:alltoall_base");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& /*opts*/) {
    DISTRO_CHECK(static_cast<int>(input_tensors.size()) == getSize() &&
                 static_cast<int>(output_tensors.size()) == getSize(),
                 "alltoall: list size != world");
    if (getSize() == 1) {
        output_tensors[0].copy_(input_tensors[0]);
        return make_completed(c10d::OpType::ALLTOALL,
                              output_tensors, "mccl:alltoall");
    }
    try {
        std::vector<at::Tensor> in_cpu(getSize()), out_cpu(getSize());
        for (int p = 0; p < getSize(); ++p) {
            in_cpu[p]  = input_tensors[p].contiguous().cpu();
            out_cpu[p] = at::empty(output_tensors[p].sizes(),
                                   output_tensors[p].options().device(at::kCPU));
        }
        out_cpu[getRank()].copy_(in_cpu[getRank()]);

        std::vector<std::thread> ts;
        ts.reserve(getSize() - 1);
        for (int p = 0; p < getSize(); ++p) {
            if (p == getRank()) continue;
            ts.emplace_back([&, p] {
                mesh_->send(p, in_cpu[p].data_ptr(), in_cpu[p].nbytes());
                mesh_->recv(p, out_cpu[p].data_ptr(), out_cpu[p].nbytes());
            });
        }
        for (auto& th : ts) th.join();
        for (int p = 0; p < getSize(); ++p) output_tensors[p].copy_(out_cpu[p]);
    } catch (...) {
        return make_failed(c10d::OpType::ALLTOALL, output_tensors,
                           "mccl:alltoall", std::current_exception());
    }
    return make_completed(c10d::OpType::ALLTOALL, output_tensors, "mccl:alltoall");
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
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
    DISTRO_CHECK(dst_rank >= 0 && dst_rank < getSize() && dst_rank != getRank(),
                 "send: bad dst_rank");
    try {
        for (auto& t : tensors) {
            auto cpu = t.contiguous().cpu();
            uint32_t nbytes = static_cast<uint32_t>(cpu.nbytes());
            int32_t hdr[2] = {tag, static_cast<int32_t>(nbytes)};
            mesh_->send(dst_rank, hdr, sizeof(hdr));
            mesh_->send(dst_rank, cpu.data_ptr(), nbytes);
        }
    } catch (...) {
        return make_failed(c10d::OpType::SEND, tensors,
                           "mccl:send", std::current_exception());
    }
    return make_completed(c10d::OpType::SEND, tensors, "mccl:send");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) {
    DISTRO_CHECK(src_rank >= 0 && src_rank < getSize() && src_rank != getRank(),
                 "recv: bad src_rank");
    try {
        for (auto& t : tensors) {
            int32_t hdr[2] = {0, 0};
            mesh_->recv(src_rank, hdr, sizeof(hdr));
            DISTRO_CHECK(tag < 0 || hdr[0] == tag,
                         "recv: tag mismatch (expected " + std::to_string(tag)
                         + " got " + std::to_string(hdr[0]) + ")");
            uint32_t nbytes = static_cast<uint32_t>(hdr[1]);
            DISTRO_CHECK(nbytes == t.nbytes(),
                         "recv: size mismatch (expected " + std::to_string(t.nbytes())
                         + " got " + std::to_string(nbytes) + ")");
            auto cpu = at::empty(t.sizes(), t.options().device(at::kCPU));
            mesh_->recv(src_rank, cpu.data_ptr(), nbytes);
            t.copy_(cpu);
        }
    } catch (...) {
        return make_failed(c10d::OpType::RECV, tensors,
                           "mccl:recv", std::current_exception());
    }
    return make_completed(c10d::OpType::RECV, tensors, "mccl:recv");
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
