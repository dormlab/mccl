#pragma once

#include <torch/csrc/distributed/c10d/Types.hpp>

#include <ATen/ATen.h>

namespace mccl {

class PeerMesh;

/// Bandwidth-optimal: 2(N-1) steps, S/N bytes per step per link.
/// Requires tensor.numel() >= world.
void allreduce_ring(at::Tensor& t, c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh);

/// Latency-optimal: 2*ceil(log2 N) steps, S bytes per step per link.
/// Handles any world size including non-power-of-2.
void allreduce_tree(at::Tensor& t, c10d::ReduceOp::RedOpType op,
                    int rank, int world, PeerMesh* mesh);

} // namespace mccl
