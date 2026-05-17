#include "backend/ProcessGroupMCCL.hpp"
#include "backend/WorkMCCL.hpp"
#include "backend/PeerMesh.hpp"
#include "backend/Progress.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/MetalKernels.hpp"
#include "runtime/Rendezvous.hpp"
#include "common/Errors.hpp"

#include <ATen/ATen.h>

#include <cstring>
#include <functional>
#include <string>
#include <thread>

namespace mccl {

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

c10::intrusive_ptr<c10d::Work> async_submit_(
    Progress& engine,
    c10d::OpType op,
    std::vector<at::Tensor> outputs,
    const char* title,
    std::function<void()> body,
    std::function<void()> sync_cb = {}) {
    auto work = c10::make_intrusive<WorkMCCL>(op, std::move(outputs), title);
    if (sync_cb) work->setSyncCallback(std::move(sync_cb));
    engine.submit([work, body = std::move(body)]() {
        try {
            body();
            work->finish();
        } catch (...) {
            work->finishWithException(std::current_exception());
        }
    });
    return work;
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
    metal_kernels_init();
    if (size > 1) {
        rendezvous_ = std::make_unique<Rendezvous>(
            store_, rank, size, opts_.timeout);
        mesh_ = std::make_unique<PeerMesh>(
            store_, rank, size, opts_.timeout);
        engine_ = std::make_unique<Progress>();
    }
}

ProcessGroupMCCL::~ProcessGroupMCCL() = default;

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce");
    }
    DISTRO_CHECK(!tensors.empty(), "allreduce: empty tensor list");
    auto op = opts.reduceOp;
    auto tensors_copy = tensors;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::ALLREDUCE, tensors,
        "mccl:allreduce", [mesh, rank, world, op, tensors_copy]() mutable {
        // Make sure any pending MPS compute writing into the input tensors is
        // committed to their MTLBuffers before we read their bytes.
        mps_event_sync();

        for (auto& t : tensors_copy) {
            DISTRO_CHECK(t.is_contiguous(),
                         "allreduce: input must be contiguous");

            auto send_sb = stage_for_send_nosync(t);
            size_t nbytes = send_sb.nbytes;

            std::vector<at::Tensor> peer_tmp(world);
            std::vector<std::vector<uint8_t>> recv_bufs(world);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                peer_tmp[p] = at::empty_like(t);
                recv_bufs[p].resize(nbytes);
            }

            std::vector<std::thread> ts;
            ts.reserve(world - 1);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                ts.emplace_back([mesh, p, send_sb, nbytes, &recv_bufs] {
                    mesh->send(p, send_sb.data, nbytes);
                    mesh->recv(p, recv_bufs[p].data(), nbytes);
                });
            }
            for (auto& th : ts) th.join();

            metal_begin_batch("mccl:allreduce");
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                unstage_from_recv(peer_tmp[p], recv_bufs[p].data(), nbytes);
                metal_reduce_op(t, peer_tmp[p], op);
            }
            if (op == c10d::ReduceOp::AVG) {
                metal_scale_inplace(t, 1.0 / static_cast<double>(world));
            }
            metal_end_batch();
        }
    },
    /*sync_cb=*/[]() { metal_sync(); });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
    if (getSize() == 1) {
        return make_completed(c10d::OpType::BROADCAST, tensors, "mccl:broadcast");
    }
    DISTRO_CHECK(!tensors.empty(), "broadcast: empty tensor list");
    auto tensors_copy = tensors;
    int root = opts.rootRank;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::BROADCAST, tensors,
        "mccl:broadcast", [mesh, rank, world, root, tensors_copy]() mutable {
        if (rank == root) mps_event_sync();
        for (auto& t : tensors_copy) {
            DISTRO_CHECK(t.is_contiguous(),
                         "broadcast: tensor must be contiguous");
            if (rank == root) {
                auto send_sb = stage_for_send_nosync(t);
                size_t nbytes = send_sb.nbytes;
                std::vector<std::thread> ts;
                ts.reserve(world - 1);
                for (int p = 0; p < world; ++p) {
                    if (p == root) continue;
                    ts.emplace_back([mesh, p, send_sb, nbytes] {
                        mesh->send(p, send_sb.data, nbytes);
                    });
                }
                for (auto& th : ts) th.join();
            } else {
                size_t nbytes = static_cast<size_t>(t.nbytes());
                std::vector<uint8_t> buf(nbytes);
                mesh->recv(root, buf.data(), nbytes);
                unstage_from_recv(t, buf.data(), nbytes);
            }
        }
    },
    /*sync_cb=*/[]() { metal_sync(); });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& /*opts*/) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_ALLGATHER_BASE,
                              {output_tensor}, "mccl:_allgather_base");
    }
    at::Tensor out_copy = output_tensor;
    at::Tensor in_copy = input_tensor;
    DISTRO_CHECK(in_copy.is_contiguous() && out_copy.is_contiguous(),
                 "_allgather_base: input/output must be contiguous");
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::_ALLGATHER_BASE,
        {output_tensor}, "mccl:_allgather_base",
        [mesh, rank, world, out_copy, in_copy]() mutable {
        mps_event_sync();
        auto send_sb = stage_for_send_nosync(in_copy);
        size_t chunk = send_sb.nbytes;
        size_t total = static_cast<size_t>(out_copy.nbytes());
        DISTRO_CHECK(total == chunk * world,
                     "_allgather_base: output size != input * world");

        std::vector<uint8_t> staging(total);
        auto* dst = staging.data();
        std::memcpy(dst + rank * chunk, send_sb.data, chunk);

        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            ts.emplace_back([mesh, p, send_sb, chunk, dst] {
                mesh->send(p, send_sb.data, chunk);
                mesh->recv(p, dst + p * chunk, chunk);
            });
        }
        for (auto& th : ts) th.join();
        unstage_from_recv(out_copy, staging.data(), total);
    },
    /*sync_cb=*/[]() { metal_sync(); });
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
    auto outs_copy = output_tensors;
    auto ins_copy = input_tensors;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::ALLGATHER, input_tensors,
        "mccl:allgather", [mesh, rank, world, outs_copy, ins_copy]() mutable {
        DISTRO_CHECK(ins_copy.size() == outs_copy.size(),
                     "allgather: input/output count mismatch");
        for (size_t i = 0; i < ins_copy.size(); ++i) {
            auto& in = ins_copy[i];
            auto& outs = outs_copy[i];
            DISTRO_CHECK(static_cast<int>(outs.size()) == world,
                         "allgather: output list size != world");
            auto in_cpu = in.contiguous().cpu();
            size_t chunk = in_cpu.nbytes();
            std::vector<at::Tensor> peer_cpu(world);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                peer_cpu[p] = at::empty(outs[p].sizes(),
                                        outs[p].options().device(at::kCPU));
            }
            std::vector<std::thread> ts;
            ts.reserve(world - 1);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                ts.emplace_back([mesh, p, &in_cpu, chunk, &peer_cpu] {
                    mesh->send(p, in_cpu.data_ptr(), chunk);
                    mesh->recv(p, peer_cpu[p].data_ptr(), chunk);
                });
            }
            for (auto& th : ts) th.join();
            outs[rank].copy_(in_cpu);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                outs[p].copy_(peer_cpu[p]);
            }
        }
    });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::_reduce_scatter_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::ReduceScatterOptions& opts) {
    if (getSize() == 1) {
        output_tensor.copy_(input_tensor);
        return make_completed(c10d::OpType::_REDUCE_SCATTER_BASE,
                              {output_tensor}, "mccl:_reduce_scatter_base");
    }
    auto op = opts.reduceOp;
    at::Tensor out_copy = output_tensor;
    at::Tensor in_copy = input_tensor;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::_REDUCE_SCATTER_BASE,
        {output_tensor}, "mccl:_reduce_scatter_base",
        [mesh, rank, world, op, out_copy, in_copy]() mutable {
        auto in_cpu = in_copy.contiguous().cpu();
        size_t chunk = out_copy.nbytes();
        DISTRO_CHECK(in_cpu.nbytes() == chunk * world,
                     "_reduce_scatter_base: input size != output * world");
        auto* src = static_cast<uint8_t*>(in_cpu.data_ptr());
        auto acc = at::empty(out_copy.sizes(),
                             out_copy.options().device(at::kCPU));
        std::memcpy(acc.data_ptr(), src + rank * chunk, chunk);
        std::vector<at::Tensor> recvbufs(world);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            recvbufs[p] = at::empty(out_copy.sizes(),
                                    out_copy.options().device(at::kCPU));
        }
        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            ts.emplace_back([mesh, p, src, chunk, &recvbufs] {
                mesh->send(p, src + p * chunk, chunk);
                mesh->recv(p, recvbufs[p].data_ptr(), chunk);
            });
        }
        for (auto& th : ts) th.join();
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            cpu_reduce_(acc, recvbufs[p], op);
        }
        if (op == c10d::ReduceOp::AVG) acc.div_(world);
        out_copy.copy_(acc);
    });
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
    auto op = opts.reduceOp;
    auto outs_copy = output_tensors;
    auto ins_copy = input_tensors;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::REDUCE_SCATTER, output_tensors,
        "mccl:reduce_scatter",
        [mesh, rank, world, op, outs_copy, ins_copy]() mutable {
        DISTRO_CHECK(outs_copy.size() == ins_copy.size(),
                     "reduce_scatter: count mismatch");
        for (size_t i = 0; i < outs_copy.size(); ++i) {
            auto& outs = outs_copy[i];
            auto& ins = ins_copy[i];
            DISTRO_CHECK(static_cast<int>(ins.size()) == world,
                         "reduce_scatter: input list size != world");
            std::vector<at::Tensor> ins_cpu(world);
            for (int p = 0; p < world; ++p) ins_cpu[p] = ins[p].contiguous().cpu();
            size_t chunk = ins_cpu[0].nbytes();
            auto acc = ins_cpu[rank].clone();
            std::vector<at::Tensor> recvbufs(world);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                recvbufs[p] = at::empty_like(ins_cpu[rank]);
            }
            std::vector<std::thread> ts;
            ts.reserve(world - 1);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                ts.emplace_back([mesh, p, &ins_cpu, chunk, &recvbufs] {
                    mesh->send(p, ins_cpu[p].data_ptr(), chunk);
                    mesh->recv(p, recvbufs[p].data_ptr(), chunk);
                });
            }
            for (auto& th : ts) th.join();
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                cpu_reduce_(acc, recvbufs[p], op);
            }
            if (op == c10d::ReduceOp::AVG) acc.div_(world);
            outs.copy_(acc);
        }
    });
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
    at::Tensor out_copy = output_tensor;
    at::Tensor in_copy = input_tensor;
    auto out_splits_copy = output_split_sizes;
    auto in_splits_copy = input_split_sizes;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::ALLTOALL_BASE,
        {output_tensor}, "mccl:alltoall_base",
        [mesh, rank, world, out_copy, in_copy, out_splits_copy, in_splits_copy]() mutable {
        auto& output_tensor = out_copy;
        auto& input_tensor = in_copy;
        auto& output_split_sizes = out_splits_copy;
        auto& input_split_sizes = in_splits_copy;
        auto in_cpu = input_tensor.contiguous().cpu();
        auto out_cpu = at::empty(output_tensor.sizes(),
                                 output_tensor.options().device(at::kCPU));
        size_t esize = static_cast<size_t>(in_cpu.element_size());
        DISTRO_CHECK(esize == static_cast<size_t>(out_cpu.element_size()),
                     "alltoall_base: dtype mismatch");

        std::vector<size_t> in_off(world + 1, 0), out_off(world + 1, 0);
        if (input_split_sizes.empty()) {
            size_t per = in_cpu.numel() / world * esize;
            DISTRO_CHECK(per * world == in_cpu.nbytes(),
                         "alltoall_base: input not divisible by world");
            for (int p = 0; p < world; ++p) in_off[p + 1] = in_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(input_split_sizes.size()) == world,
                         "alltoall_base: input_split_sizes wrong length");
            for (int p = 0; p < world; ++p)
                in_off[p + 1] = in_off[p] + input_split_sizes[p] * esize;
        }
        if (output_split_sizes.empty()) {
            size_t per = out_cpu.numel() / world * esize;
            DISTRO_CHECK(per * world == out_cpu.nbytes(),
                         "alltoall_base: output not divisible by world");
            for (int p = 0; p < world; ++p) out_off[p + 1] = out_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(output_split_sizes.size()) == world,
                         "alltoall_base: output_split_sizes wrong length");
            for (int p = 0; p < world; ++p)
                out_off[p + 1] = out_off[p] + output_split_sizes[p] * esize;
        }

        auto* src = static_cast<uint8_t*>(in_cpu.data_ptr());
        auto* dst = static_cast<uint8_t*>(out_cpu.data_ptr());

        size_t self_in_len  = in_off[rank + 1]  - in_off[rank];
        size_t self_out_len = out_off[rank + 1] - out_off[rank];
        DISTRO_CHECK(self_in_len == self_out_len,
                     "alltoall_base: self chunk size mismatch");
        std::memcpy(dst + out_off[rank], src + in_off[rank], self_in_len);

        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            size_t slen = in_off[p + 1]  - in_off[p];
            size_t rlen = out_off[p + 1] - out_off[p];
            if (slen == 0 && rlen == 0) continue;
            ts.emplace_back([mesh, p, src, dst, &in_off, &out_off, slen, rlen] {
                if (slen) mesh->send(p, src + in_off[p], slen);
                if (rlen) mesh->recv(p, dst + out_off[p], rlen);
            });
        }
        for (auto& th : ts) th.join();
        output_tensor.copy_(out_cpu);
    });
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
    auto outs_copy = output_tensors;
    auto ins_copy = input_tensors;
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::ALLTOALL, output_tensors,
        "mccl:alltoall", [mesh, rank, world, outs_copy, ins_copy]() mutable {
        std::vector<at::Tensor> in_cpu(world), out_cpu(world);
        for (int p = 0; p < world; ++p) {
            in_cpu[p]  = ins_copy[p].contiguous().cpu();
            out_cpu[p] = at::empty(outs_copy[p].sizes(),
                                   outs_copy[p].options().device(at::kCPU));
        }
        out_cpu[rank].copy_(in_cpu[rank]);

        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            ts.emplace_back([mesh, p, &in_cpu, &out_cpu] {
                mesh->send(p, in_cpu[p].data_ptr(), in_cpu[p].nbytes());
                mesh->recv(p, out_cpu[p].data_ptr(), out_cpu[p].nbytes());
            });
        }
        for (auto& th : ts) th.join();
        for (int p = 0; p < world; ++p) outs_copy[p].copy_(out_cpu[p]);
    });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::barrier(
    const c10d::BarrierOptions& /*opts*/) {
    if (getSize() == 1 || !rendezvous_) {
        return make_completed(c10d::OpType::BARRIER, {}, "mccl:barrier");
    }
    Rendezvous* rv = rendezvous_.get();
    std::string tag = "pg_barrier_" + std::to_string(next_seq_());
    return async_submit_(*engine_, c10d::OpType::BARRIER, {}, "mccl:barrier",
        [rv, tag]() { rv->barrier(tag); });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
    DISTRO_CHECK(dst_rank >= 0 && dst_rank < getSize() && dst_rank != getRank(),
                 "send: bad dst_rank");
    auto tensors_copy = tensors;
    PeerMesh* mesh = mesh_.get();
    return async_submit_(*engine_, c10d::OpType::SEND, tensors, "mccl:send",
        [mesh, dst_rank, tag, tensors_copy]() mutable {
        for (auto& t : tensors_copy) {
            auto cpu = t.contiguous().cpu();
            uint32_t nbytes = static_cast<uint32_t>(cpu.nbytes());
            int32_t hdr[2] = {tag, static_cast<int32_t>(nbytes)};
            mesh->send(dst_rank, hdr, sizeof(hdr));
            mesh->send(dst_rank, cpu.data_ptr(), nbytes);
        }
    });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) {
    DISTRO_CHECK(src_rank >= 0 && src_rank < getSize() && src_rank != getRank(),
                 "recv: bad src_rank");
    auto tensors_copy = tensors;
    PeerMesh* mesh = mesh_.get();
    return async_submit_(*engine_, c10d::OpType::RECV, tensors, "mccl:recv",
        [mesh, src_rank, tag, tensors_copy]() mutable {
        for (auto& t : tensors_copy) {
            int32_t hdr[2] = {0, 0};
            mesh->recv(src_rank, hdr, sizeof(hdr));
            DISTRO_CHECK(tag < 0 || hdr[0] == tag,
                         "recv: tag mismatch (expected " + std::to_string(tag)
                         + " got " + std::to_string(hdr[0]) + ")");
            uint32_t nbytes = static_cast<uint32_t>(hdr[1]);
            DISTRO_CHECK(nbytes == t.nbytes(),
                         "recv: size mismatch (expected " + std::to_string(t.nbytes())
                         + " got " + std::to_string(nbytes) + ")");
            auto cpu = at::empty(t.sizes(), t.options().device(at::kCPU));
            mesh->recv(src_rank, cpu.data_ptr(), nbytes);
            t.copy_(cpu);
        }
    });
}

c10::intrusive_ptr<c10d::Backend> ProcessGroupMCCL::create(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank, int size,
    const std::chrono::duration<float>& timeout) {
    Options opts;
    opts.timeout = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
    return c10::make_intrusive<ProcessGroupMCCL>(store, rank, size, opts);
}

} // namespace mccl
