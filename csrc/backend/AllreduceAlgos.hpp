#pragma once

#include <torch/csrc/distributed/c10d/Types.hpp>

#include <ATen/ATen.h>

#include "metal/MPSInterop.hpp"

namespace mccl {

class PeerMesh;

/// Inputs all pre-extracted on the calling thread (torch's main MPS-aware
/// thread): the tensor's MTLBuffer view, the recv-scratch tensor + view,
/// scalar dtype, and element count. The algorithm body never calls into
/// torch's MPS allocator from the engine thread — that path deadlocks.
struct AllreduceArgs {
    MPSBufferView t_view;          // view into the user's tensor
    at::Tensor recv_scratch;       // engine-allocated keeps storage alive
    MPSBufferView recv_view;       // view into recv_scratch
    at::ScalarType dtype;
    uint32_t numel;                // element count of t
};

/// Bandwidth-optimal: 2(N-1) steps, S/N bytes per step per link.
/// Requires numel >= world.
void allreduce_ring(const AllreduceArgs& args,
                    c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh);

/// Latency-optimal: 2*ceil(log2 N) steps, S bytes per step per link.
/// Handles any world size including non-power-of-2.
void allreduce_tree(const AllreduceArgs& args,
                    c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh);

/// Full-mesh butterfly: 2 phases (reduce-scatter then allgather), each
/// rank sends to all (world-1) peers in parallel per phase. For N<=4 on
/// a TB mesh this halves the sequentialised step count vs ring.
/// recv_scratch must be sized to (world-1) * per_chunk_elems for this algo.
void allreduce_butterfly(const AllreduceArgs& args,
                         c10d::ReduceOp::RedOpType op,
                         int rank, int world, PeerMesh* mesh);

} // namespace mccl
