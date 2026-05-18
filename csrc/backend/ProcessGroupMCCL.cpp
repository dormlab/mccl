#include "backend/ProcessGroupMCCL.hpp"
#include "backend/WorkMCCL.hpp"
#include "backend/PeerMesh.hpp"
#include "backend/Progress.hpp"
#include "backend/AllreduceAlgos.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/MetalKernels.hpp"
#include "metal/EventSync.hpp"
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
    // commit_mps_and_signal touches torch's MPS dispatch queue via
    // dispatch_sync; call it on the issuing thread before handing the
    // body off, then wait for the signal on the engine thread. Doing
    // both inside the engine task can deadlock because the engine
    // thread doesn't own torch's per-thread Metal command-buffer state.
    uint64_t sync_val = mps_event_sync_nonblocking();
    auto work = c10::make_intrusive<WorkMCCL>(op, std::move(outputs), title);
    if (sync_cb) work->setSyncCallback(std::move(sync_cb));
    engine.submit([work, body = std::move(body), sync_val]() {
        try {
            if (sync_val > 0) wait_for_mps(sync_val);
            body();
            work->finish();
        } catch (...) {
            work->finishWithException(std::current_exception());
        }
    });
    return work;
}

void check_mps_(const at::Tensor& t, const char* op) {
    DISTRO_CHECK(t.is_mps(),
                 std::string(op) + ": MCCL is MPS-only; got "
                 + std::string(c10::DeviceTypeName(t.device().type()))
                 + " tensor");
    DISTRO_CHECK(t.is_contiguous(),
                 std::string(op) + ": tensor must be contiguous");
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
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();

    // Optional bf16 wire compression. When MCCL_WIRE_DTYPE=bf16 and the
    // input is fp32, we run the entire ring/tree in bf16 — half the
    // bytes on the wire, half the bytes through the Metal reduce kernel.
    // The fp32→bf16 / bf16→fp32 casts are MPS ops on torch's stream and
    // are amortised by the network savings on Thunderbolt-bound runs.
    bool wire_bf16 = [] {
        const char* w = std::getenv("MCCL_WIRE_DTYPE");
        return w && std::string(w) == "bf16";
    }();

    std::vector<AllreduceArgs> argv;
    std::vector<std::string> algv;
    std::vector<at::Tensor> work_tensors;  // keeps bf16 scratch alive
    argv.reserve(tensors.size());
    algv.reserve(tensors.size());
    work_tensors.reserve(tensors.size());
    for (auto& t : tensors) {
        check_mps_(t, "allreduce");

        at::Tensor wt = (wire_bf16 && t.scalar_type() == at::kFloat)
                            ? t.to(at::kBFloat16) : t;
        size_t bytes = static_cast<size_t>(wt.nbytes());

        const char* env_alg = std::getenv("MCCL_ALLREDUCE_ALGO");
        std::string alg = env_alg ? env_alg : "auto";
        if (alg == "auto") {
            size_t tree_below = 256ull * 1024;
            if (const char* e = std::getenv("MCCL_TREE_BELOW"))
                tree_below = std::strtoull(e, nullptr, 10);
            alg = (bytes <= tree_below) ? "tree" : "ring";
        }
        if (alg == "ring" && wt.numel() < world) alg = "tree";

        AllreduceArgs a;
        a.t_view  = extract_mps_buffer(wt);
        a.dtype   = wt.scalar_type();
        a.numel   = static_cast<uint32_t>(wt.numel());

        if (alg == "ring") {
            int64_t per = (wt.numel() + world - 1) / world;
            a.recv_scratch = at::empty({per}, wt.options());
        } else {
            a.recv_scratch = at::empty_like(wt);
        }
        a.recv_view = extract_mps_buffer(a.recv_scratch);

        argv.push_back(std::move(a));
        algv.push_back(std::move(alg));
        work_tensors.push_back(std::move(wt));
    }

    // Inline-sync. Async path was instrumented end-to-end with
    // MCCL_COMMIT wrappers around every [cmd commit] in our code;
    // across crashing async runs we saw zero double-commits at our
    // sites, so the offender is inside torch's MPSStream::flush ->
    // [_commandBuffer commit] interacting with MPSGraph mid-encode
    // on the same buffer. Cannot fix from outside pytorch. Async
    // works in ~1/5 runs (timing-dependent). Leaving the wiring in
    // place — see commit_mps_and_signal / wait_for_mps in
    // EventSync.mm.
    mps_stream_sync();
    for (size_t i = 0; i < argv.size(); ++i) {
        if (algv[i] == "ring") allreduce_ring(argv[i], op, rank, world, mesh);
        else                    allreduce_tree(argv[i], op, rank, world, mesh);
    }
    mccl_queue_drain();
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (work_tensors[i].data_ptr() != tensors[i].data_ptr()) {
            tensors[i].copy_(work_tensors[i].to(tensors[i].scalar_type()));
        }
    }
    if (wire_bf16) mps_stream_sync();
    return make_completed(c10d::OpType::ALLREDUCE, tensors, "mccl:allreduce");
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
        
        for (auto& t : tensors_copy) {
            check_mps_(t, "broadcast");
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
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
    PeerMesh* mesh = mesh_.get();
    int rank = getRank();
    int world = getSize();
    return async_submit_(*engine_, c10d::OpType::_ALLGATHER_BASE,
        {output_tensor}, "mccl:_allgather_base",
        [mesh, rank, world, out_copy, in_copy]() mutable {
        check_mps_(in_copy, "_allgather_base");
        check_mps_(out_copy, "_allgather_base");
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
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
            check_mps_(in, "allgather");
            for (auto& o : outs) check_mps_(o, "allgather");

            auto send_sb = stage_for_send_nosync(in);
            size_t chunk = send_sb.nbytes;
            std::vector<std::vector<uint8_t>> peer_buf(world);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                peer_buf[p].resize(chunk);
            }
            std::vector<std::thread> ts;
            ts.reserve(world - 1);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                ts.emplace_back([mesh, p, send_sb, chunk, &peer_buf] {
                    mesh->send(p, send_sb.data, chunk);
                    mesh->recv(p, peer_buf[p].data(), chunk);
                });
            }
            for (auto& th : ts) th.join();

            unstage_from_recv(outs[rank], send_sb.data, chunk);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                unstage_from_recv(outs[p], peer_buf[p].data(), chunk);
            }
        }
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
        check_mps_(in_copy, "_reduce_scatter_base");
        check_mps_(out_copy, "_reduce_scatter_base");
        auto send_sb = stage_for_send_nosync(in_copy);
        size_t chunk = static_cast<size_t>(out_copy.nbytes());
        DISTRO_CHECK(send_sb.nbytes == chunk * world,
                     "_reduce_scatter_base: input size != output * world");

        // Seed output with our own share; we then reduce peers' shares into it.
        const auto* src = static_cast<const uint8_t*>(send_sb.data);
        unstage_from_recv(out_copy, src + rank * chunk, chunk);

        std::vector<at::Tensor> peer_tmp(world);
        std::vector<std::vector<uint8_t>> recv_bufs(world);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            peer_tmp[p] = at::empty_like(out_copy);
            recv_bufs[p].resize(chunk);
        }

        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            ts.emplace_back([mesh, p, src, chunk, &recv_bufs] {
                mesh->send(p, src + p * chunk, chunk);
                mesh->recv(p, recv_bufs[p].data(), chunk);
            });
        }
        for (auto& th : ts) th.join();

        metal_begin_batch("mccl:_reduce_scatter_base");
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            unstage_from_recv(peer_tmp[p], recv_bufs[p].data(), chunk);
            metal_reduce_op(out_copy, peer_tmp[p], op);
        }
        if (op == c10d::ReduceOp::AVG) {
            metal_scale_inplace(out_copy, 1.0 / static_cast<double>(world));
        }
        metal_end_batch();
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
            auto& out = outs_copy[i];
            auto& ins = ins_copy[i];
            DISTRO_CHECK(static_cast<int>(ins.size()) == world,
                         "reduce_scatter: input list size != world");

            check_mps_(out, "reduce_scatter");
            std::vector<StagingBuffer> send_sbs(world);
            for (int p = 0; p < world; ++p) {
                check_mps_(ins[p], "reduce_scatter");
                send_sbs[p] = stage_for_send_nosync(ins[p]);
            }
            size_t chunk = send_sbs[0].nbytes;
            DISTRO_CHECK(static_cast<size_t>(out.nbytes()) == chunk,
                         "reduce_scatter: output size != input chunk size");

            unstage_from_recv(out, send_sbs[rank].data, chunk);

            std::vector<at::Tensor> peer_tmp(world);
            std::vector<std::vector<uint8_t>> recv_bufs(world);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                peer_tmp[p] = at::empty_like(out);
                recv_bufs[p].resize(chunk);
            }

            std::vector<std::thread> ts;
            ts.reserve(world - 1);
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                ts.emplace_back([mesh, p, sb = send_sbs[p], chunk, &recv_bufs] {
                    mesh->send(p, sb.data, chunk);
                    mesh->recv(p, recv_bufs[p].data(), chunk);
                });
            }
            for (auto& th : ts) th.join();

            metal_begin_batch("mccl:reduce_scatter");
            for (int p = 0; p < world; ++p) {
                if (p == rank) continue;
                unstage_from_recv(peer_tmp[p], recv_bufs[p].data(), chunk);
                metal_reduce_op(out, peer_tmp[p], op);
            }
            if (op == c10d::ReduceOp::AVG) {
                metal_scale_inplace(out, 1.0 / static_cast<double>(world));
            }
            metal_end_batch();
        }
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
        check_mps_(input_tensor, "alltoall_base");
        check_mps_(output_tensor, "alltoall_base");

        auto send_sb = stage_for_send_nosync(input_tensor);
        size_t esize = static_cast<size_t>(input_tensor.element_size());
        DISTRO_CHECK(esize == static_cast<size_t>(output_tensor.element_size()),
                     "alltoall_base: dtype mismatch");
        size_t out_bytes = static_cast<size_t>(output_tensor.nbytes());

        std::vector<size_t> in_off(world + 1, 0), out_off(world + 1, 0);
        if (input_split_sizes.empty()) {
            size_t per = static_cast<size_t>(input_tensor.numel()) / world * esize;
            DISTRO_CHECK(per * world == send_sb.nbytes,
                         "alltoall_base: input not divisible by world");
            for (int p = 0; p < world; ++p) in_off[p + 1] = in_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(input_split_sizes.size()) == world,
                         "alltoall_base: input_split_sizes wrong length");
            for (int p = 0; p < world; ++p)
                in_off[p + 1] = in_off[p] + input_split_sizes[p] * esize;
        }
        if (output_split_sizes.empty()) {
            size_t per = static_cast<size_t>(output_tensor.numel()) / world * esize;
            DISTRO_CHECK(per * world == out_bytes,
                         "alltoall_base: output not divisible by world");
            for (int p = 0; p < world; ++p) out_off[p + 1] = out_off[p] + per;
        } else {
            DISTRO_CHECK(static_cast<int>(output_split_sizes.size()) == world,
                         "alltoall_base: output_split_sizes wrong length");
            for (int p = 0; p < world; ++p)
                out_off[p + 1] = out_off[p] + output_split_sizes[p] * esize;
        }

        const auto* src = static_cast<const uint8_t*>(send_sb.data);
        std::vector<uint8_t> dst_buf(out_bytes);
        auto* dst = dst_buf.data();

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
        unstage_from_recv(output_tensor, dst_buf.data(), out_bytes);
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
        std::vector<StagingBuffer> send_sbs(world);
        std::vector<std::vector<uint8_t>> recv_bufs(world);
        for (int p = 0; p < world; ++p) {
            check_mps_(ins_copy[p], "alltoall");
            check_mps_(outs_copy[p], "alltoall");
            send_sbs[p] = stage_for_send_nosync(ins_copy[p]);
            if (p == rank) continue;
            recv_bufs[p].resize(static_cast<size_t>(outs_copy[p].nbytes()));
        }

        unstage_from_recv(outs_copy[rank], send_sbs[rank].data,
                          send_sbs[rank].nbytes);

        std::vector<std::thread> ts;
        ts.reserve(world - 1);
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            ts.emplace_back([mesh, p, sb = send_sbs[p], &recv_bufs] {
                mesh->send(p, sb.data, sb.nbytes);
                mesh->recv(p, recv_bufs[p].data(), recv_bufs[p].size());
            });
        }
        for (auto& th : ts) th.join();

        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            unstage_from_recv(outs_copy[p], recv_bufs[p].data(),
                              recv_bufs[p].size());
        }
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
            check_mps_(t, "send");
            auto sb = stage_for_send_nosync(t);
            uint32_t nbytes = static_cast<uint32_t>(sb.nbytes);
            int32_t hdr[2] = {tag, static_cast<int32_t>(nbytes)};
            mesh->send(dst_rank, hdr, sizeof(hdr));
            mesh->send(dst_rank, sb.data, nbytes);
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
            check_mps_(t, "recv");
            int32_t hdr[2] = {0, 0};
            mesh->recv(src_rank, hdr, sizeof(hdr));
            DISTRO_CHECK(tag < 0 || hdr[0] == tag,
                         "recv: tag mismatch (expected " + std::to_string(tag)
                         + " got " + std::to_string(hdr[0]) + ")");
            uint32_t nbytes = static_cast<uint32_t>(hdr[1]);
            DISTRO_CHECK(nbytes == t.nbytes(),
                         "recv: size mismatch (expected " + std::to_string(t.nbytes())
                         + " got " + std::to_string(nbytes) + ")");
            std::vector<uint8_t> buf(nbytes);
            mesh->recv(src_rank, buf.data(), nbytes);
            unstage_from_recv(t, buf.data(), nbytes);
        }
    },
    /*sync_cb=*/[]() { mccl_queue_drain(); });
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
