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

std::vector<ChunkInfo> chunkize(const at::Tensor& t, int world) {
    int64_t numel = t.numel();
    int64_t per = (numel + world - 1) / world;
    size_t esize = static_cast<size_t>(t.element_size());
    std::vector<ChunkInfo> chunks(world);
    for (int i = 0; i < world; ++i) {
        int64_t s = std::min<int64_t>(static_cast<int64_t>(i) * per, numel);
        int64_t e = std::min<int64_t>(static_cast<int64_t>(i + 1) * per, numel);
        chunks[i].elem_start = s;
        chunks[i].elem_count = e - s;
        chunks[i].byte_off = static_cast<size_t>(s) * esize;
        chunks[i].byte_len = static_cast<size_t>(e - s) * esize;
    }
    return chunks;
}

} // namespace

void allreduce_ring(at::Tensor& t, c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh) {
    DISTRO_CHECK(t.numel() >= static_cast<int64_t>(world),
                 "ring allreduce: tensor numel must be >= world");

    auto flat = t.view(-1);
    auto chunks = chunkize(flat, world);
    int next = (rank + 1) % world;
    int prev = (rank - 1 + world) % world;

    auto stage = stage_for_send_nosync(t);
    auto* base = static_cast<uint8_t*>(stage.data);

    size_t max_bytes = 0;
    int64_t max_elems = 0;
    for (auto& c : chunks) {
        max_bytes = std::max(max_bytes, c.byte_len);
        max_elems = std::max(max_elems, c.elem_count);
    }
    auto recv_buf = at::empty({max_elems}, flat.options());

    mps_event_sync();

    // Phase 1 — reduce-scatter (N-1 steps).
    for (int k = 0; k < world - 1; ++k) {
        int send_idx = (rank - k + world) % world;
        int recv_idx = (rank - k - 1 + world) % world;
        const auto& s = chunks[send_idx];
        const auto& r = chunks[recv_idx];

        std::thread st([&]{ mesh->send(next, base + s.byte_off, s.byte_len); });
        mesh->recv(prev, recv_buf.data_ptr(), r.byte_len);
        st.join();

        auto t_chunk = flat.narrow(0, r.elem_start, r.elem_count);
        auto rb = recv_buf.narrow(0, 0, r.elem_count);
        metal_begin_batch("ring-rs");
        metal_reduce_op(t_chunk, rb, op);
        metal_end_batch();
        metal_sync();
    }

    // Phase 2 — allgather (N-1 steps).
    for (int k = 0; k < world - 1; ++k) {
        int send_idx = (rank - k + 1 + world) % world;
        int recv_idx = (rank - k + world) % world;
        const auto& s = chunks[send_idx];
        const auto& r = chunks[recv_idx];

        std::thread st([&]{ mesh->send(next, base + s.byte_off, s.byte_len); });
        mesh->recv(prev, recv_buf.data_ptr(), r.byte_len);
        st.join();

        auto t_chunk = flat.narrow(0, r.elem_start, r.elem_count);
        auto rb = recv_buf.narrow(0, 0, r.elem_count);
        t_chunk.copy_(rb);
    }

    if (op == c10d::ReduceOp::AVG) {
        metal_begin_batch("ring-avg");
        metal_scale_inplace(t, 1.0 / static_cast<double>(world));
        metal_end_batch();
    }
}

void allreduce_tree(at::Tensor& t, c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh) {
    auto stage = stage_for_send_nosync(t);
    auto recv_buf = at::empty_like(t);
    size_t nbytes = static_cast<size_t>(t.nbytes());

    mps_event_sync();

    // Reduce phase — binomial fold to rank 0.
    bool active = true;
    for (int mask = 1; mask < world; mask <<= 1) {
        if (!active) break;
        if (rank & mask) {
            int dst = rank - mask;
            mesh->send(dst, stage.data, nbytes);
            active = false;
        } else {
            int src = rank + mask;
            if (src < world) {
                mesh->recv(src, recv_buf.data_ptr(), nbytes);
                metal_begin_batch("tree-up");
                metal_reduce_op(t, recv_buf, op);
                metal_end_batch();
                metal_sync();
            }
        }
    }

    // Broadcast phase — rank 0 outward, mask high to low.
    int top_mask = 1;
    while (top_mask * 2 < world) top_mask <<= 1;

    for (int mask = top_mask; mask >= 1; mask >>= 1) {
        int twomask = mask << 1;
        if ((rank % twomask) == 0 && (rank + mask) < world) {
            mesh->send(rank + mask, stage.data, nbytes);
        } else if ((rank % twomask) == mask) {
            mesh->recv(rank - mask, t.data_ptr(), nbytes);
        }
    }

    if (op == c10d::ReduceOp::AVG) {
        metal_begin_batch("tree-avg");
        metal_scale_inplace(t, 1.0 / static_cast<double>(world));
        metal_end_batch();
    }
}

} // namespace mccl
