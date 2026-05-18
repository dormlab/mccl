#include "backend/AllreduceAlgos.hpp"
#include "backend/PeerMesh.hpp"
#include "metal/MPSInterop.hpp"
#include "metal/MetalKernels.hpp"
#include "common/Errors.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace mccl {

namespace {

struct ChunkInfo {
    int64_t elem_start;
    int64_t elem_count;
    size_t byte_off;
    size_t byte_len;
};

std::vector<ChunkInfo> chunkize(int64_t numel, size_t elem_size, int world) {
    int64_t per = (numel + world - 1) / world;
    std::vector<ChunkInfo> chunks(world);
    for (int i = 0; i < world; ++i) {
        int64_t s = std::min<int64_t>(static_cast<int64_t>(i) * per, numel);
        int64_t e = std::min<int64_t>(static_cast<int64_t>(i + 1) * per, numel);
        chunks[i].elem_start = s;
        chunks[i].elem_count = e - s;
        chunks[i].byte_off = static_cast<size_t>(s) * elem_size;
        chunks[i].byte_len = static_cast<size_t>(e - s) * elem_size;
    }
    return chunks;
}

MPSBufferView view_at(const MPSBufferView& base, size_t offset) {
    return MPSBufferView{
        base.mtl_buffer, base.byte_offset + offset, base.nbytes - offset,
        base.cpu_accessible,
        base.cpu_ptr ? static_cast<uint8_t*>(base.cpu_ptr) + offset : nullptr
    };
}

void* byte_ptr(const MPSBufferView& v, size_t offset = 0) {
    DISTRO_CHECK(v.cpu_ptr != nullptr,
                 "AllreduceAlgos requires shared (cpu-accessible) MTLBuffer storage");
    return static_cast<uint8_t*>(v.cpu_ptr) + offset;
}

} // namespace

void allreduce_ring(const AllreduceArgs& args,
                    c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh) {
    DISTRO_CHECK(args.numel >= static_cast<uint32_t>(world),
                 "ring allreduce: numel must be >= world");
    size_t esize = static_cast<size_t>(at::elementSize(args.dtype));
    auto chunks = chunkize(args.numel, esize, world);
    int next = (rank + 1) % world;
    int prev = (rank - 1 + world) % world;

    for (int k = 0; k < world - 1; ++k) {
        int send_idx = (rank - k + world) % world;
        int recv_idx = (rank - k - 1 + world) % world;
        const auto& s = chunks[send_idx];
        const auto& r = chunks[recv_idx];

        std::thread st([&]{ mesh->send(next, byte_ptr(args.t_view, s.byte_off), s.byte_len); });
        mesh->recv(prev, byte_ptr(args.recv_view), r.byte_len);
        st.join();

        auto dst_v = view_at(args.t_view, r.byte_off);
        metal_reduce_op_view(dst_v, args.recv_view, args.dtype,
                             static_cast<uint32_t>(r.elem_count), op);
    }

    for (int k = 0; k < world - 1; ++k) {
        int send_idx = (rank - k + 1 + world) % world;
        int recv_idx = (rank - k + world) % world;
        const auto& s = chunks[send_idx];
        const auto& r = chunks[recv_idx];

        std::thread st([&]{ mesh->send(next, byte_ptr(args.t_view, s.byte_off), s.byte_len); });
        mesh->recv(prev, byte_ptr(args.t_view, r.byte_off), r.byte_len);
        st.join();
    }

    if (op == c10d::ReduceOp::AVG) {
        metal_scale_inplace_view(args.t_view, args.dtype, args.numel,
                                 1.0 / static_cast<double>(world));
    }
}

void allreduce_tree(const AllreduceArgs& args,
                    c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh) {
    size_t nbytes = static_cast<size_t>(args.numel) *
                    static_cast<size_t>(at::elementSize(args.dtype));

    bool active = true;
    for (int mask = 1; mask < world; mask <<= 1) {
        if (!active) break;
        if (rank & mask) {
            mesh->send(rank - mask, byte_ptr(args.t_view), nbytes);
            active = false;
        } else {
            int src = rank + mask;
            if (src < world) {
                mesh->recv(src, byte_ptr(args.recv_view), nbytes);
                metal_reduce_op_view(args.t_view, args.recv_view,
                                     args.dtype, args.numel, op);
            }
        }
    }

    int top_mask = 1;
    while (top_mask * 2 < world) top_mask <<= 1;

    for (int mask = top_mask; mask >= 1; mask >>= 1) {
        int twomask = mask << 1;
        if ((rank % twomask) == 0 && (rank + mask) < world) {
            mesh->send(rank + mask, byte_ptr(args.t_view), nbytes);
        } else if ((rank % twomask) == mask) {
            mesh->recv(rank - mask, byte_ptr(args.t_view), nbytes);
        }
    }

    if (op == c10d::ReduceOp::AVG) {
        metal_scale_inplace_view(args.t_view, args.dtype, args.numel,
                                 1.0 / static_cast<double>(world));
    }
}

void allreduce_butterfly(const AllreduceArgs& args,
                         c10d::ReduceOp::RedOpType op,
                         int rank, int world, PeerMesh* mesh) {
    DISTRO_CHECK(args.numel >= static_cast<uint32_t>(world),
                 "butterfly allreduce: numel must be >= world");
    size_t esize = static_cast<size_t>(at::elementSize(args.dtype));
    auto chunks = chunkize(args.numel, esize, world);
    const auto& my_chunk = chunks[rank];

    // recv_view must hold (world-1) chunks of size my_chunk.byte_len.
    DISTRO_CHECK(args.recv_view.nbytes >=
                 static_cast<size_t>(world - 1) * my_chunk.byte_len,
                 "butterfly: recv_view too small; allocate (world-1)*per chunks");

    // Phase 1 — reduce-scatter. Each rank R sends chunks[p] to peer p
    // (for p != R), parallel across all peers. Each rank R recvs its
    // own chunk from each peer in parallel.
    {
        std::vector<std::thread> ts;
        ts.reserve(2 * (world - 1));
        int slot = 0;
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            const auto& s = chunks[p];
            ts.emplace_back([&, p, s] {
                mesh->send(p, byte_ptr(args.t_view, s.byte_off), s.byte_len);
            });
            size_t recv_off = static_cast<size_t>(slot) * my_chunk.byte_len;
            ts.emplace_back([&, p, recv_off] {
                mesh->recv(p, byte_ptr(args.recv_view, recv_off),
                           my_chunk.byte_len);
            });
            slot++;
        }
        for (auto& t : ts) t.join();
    }

    // Reduce all (world-1) received slots into the rank's own chunk.
    {
        auto dst_v = view_at(args.t_view, my_chunk.byte_off);
        for (int slot = 0; slot < world - 1; ++slot) {
            MPSBufferView src = view_at(args.recv_view,
                static_cast<size_t>(slot) * my_chunk.byte_len);
            metal_reduce_op_view(dst_v, src, args.dtype,
                                 static_cast<uint32_t>(my_chunk.elem_count), op);
        }
    }

    // Phase 2 — allgather. Each rank broadcasts its own reduced chunk
    // to all other peers in parallel; recvs the others' chunks in
    // parallel into the right offsets in the user tensor.
    {
        std::vector<std::thread> ts;
        ts.reserve(2 * (world - 1));
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            const auto& s = my_chunk;
            ts.emplace_back([&, p, s] {
                mesh->send(p, byte_ptr(args.t_view, s.byte_off), s.byte_len);
            });
            const auto& r = chunks[p];
            ts.emplace_back([&, p, r] {
                mesh->recv(p, byte_ptr(args.t_view, r.byte_off), r.byte_len);
            });
        }
        for (auto& t : ts) t.join();
    }

    if (op == c10d::ReduceOp::AVG) {
        metal_scale_inplace_view(args.t_view, args.dtype, args.numel,
                                 1.0 / static_cast<double>(world));
    }
}

} // namespace mccl
