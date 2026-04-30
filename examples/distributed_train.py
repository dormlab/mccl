#!/usr/bin/env python3
"""
distro distributed training example.

Each Mac mini runs this script with a different --node-id.
The head node (usually node 0) coordinates.

Usage (data-parallel, 3 nodes)::

    # On mini-0 (head):
    python distributed_train.py --node-id 0 --world-size 3 --mode data_parallel

    # On mini-1:
    python distributed_train.py --node-id 1 --world-size 3 --mode data_parallel

    # On mini-2:
    python distributed_train.py --node-id 2 --world-size 3 --mode data_parallel

The script:
  1. Initializes DMEM on each node
  2. Registers model parameters/gradients with the RDMA NIC
  3. Runs the training loop with gradient sync via RDMA writes (no allreduce)
  4. Reports throughput and metrics
"""

from __future__ import annotations

import argparse
import os
import time
import torch
import torch.nn as nn

import distro
from distro.distributed.tensor import DistributedTensor, ShardingSpec
from distro.distributed.sharding import DistributedModel


# ── Model ──────────────────────────────────────────────────────────────────

class DemoModel(nn.Module):
    """Small MLP for demonstration — enough params to make RDMA worthwhile."""

    def __init__(self, hidden: int = 2048, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = 1024
        for i in range(depth):
            out_dim = hidden if i < depth - 1 else 10
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data ───────────────────────────────────────────────────────────────────

def make_batch(batch_size: int, device: str = "mps"):
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


# ── Data-parallel training ─────────────────────────────────────────────────

def train_data_parallel(args):
    """Each node has a full model replica, processes a batch shard,
    then syncs gradients via RDMA writes."""

    device = torch.device("mps")
    model = DemoModel(hidden=args.hidden, depth=args.depth).to(device)
    model.train()

    # Wrap for distributed training — registers all parameters with DMEM
    dist_model = DistributedModel(
        model, node_id=args.node_id, world_size=args.world_size
    )
    dist_model.register_parameters()

    print(f"[node {args.node_id}] Parameters registered: {len(dist_model._param_regions)}")
    for name, rid in sorted(dist_model._param_regions.items()):
        param = dict(model.named_parameters())[name]
        print(f"  {name}: region={rid}, shape={list(param.shape)}, "
              f"nbytes={param.numel() * param.element_size()}")

    # In production, peer regions are exchanged via the head node.
    # For this demo, each node uses the same model structure so region
    # IDs match parameter names.  In a real deployment, the head
    # coordinates region ID exchange.
    #
    # Simulate: each peer has the same parameter shapes/regions.
    # This is a simplification — real exchange goes through the head's
    # memory catalog (RDMA-readable by all peers).
    for peer_id in range(args.world_size):
        if peer_id == args.node_id:
            continue
        peer_regions = {}
        for name, param in model.named_parameters():
            peer_regions[name] = (
                dist_model._param_regions[name],
                param.numel(),
            )
        dist_model.set_peer_regions(peer_id, peer_regions)

    print(f"[node {args.node_id}] Peers configured: "
          f"{list(dist_model._peer_param_regions.keys())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Each node gets a slice of the global batch
    per_rank_batch = args.batch_size
    global_batch = per_rank_batch * args.world_size

    print(f"[node {args.node_id}] Starting training "
          f"(global_batch={global_batch}, steps={args.steps})")

    times = []
    torch.manual_seed(42 + args.node_id)

    for step in range(args.steps):
        t0 = time.perf_counter()

        x, y = make_batch(per_rank_batch)

        optimizer.zero_grad(set_to_none=True)
        output = dist_model(x)
        loss = loss_fn(output, y)
        loss.backward()

        # Gradient sync — this is where RDMA happens
        dist_model.sync_gradients()

        optimizer.step()

        t1 = time.perf_counter()
        step_time = t1 - t0
        times.append(step_time)

        if args.node_id == 0 or step % 10 == 0:
            print(f"[node {args.node_id}] step {step}: "
                  f"loss={loss.item():.4f}, "
                  f"time={step_time*1000:.1f}ms, "
                  f"throughput={global_batch/step_time:.1f} samples/s")

    # Report
    avg_time = sum(times) / len(times)
    print(f"\n[node {args.node_id}] Complete.")
    print(f"  Avg step: {avg_time*1000:.1f}ms")
    print(f"  Throughput: {global_batch/avg_time:.1f} samples/s")
    print(f"  DMEM stats: {distro.get_stats()}")

    dist_model.deregister_parameters()


# ── Tensor-parallel training ───────────────────────────────────────────────

def train_tensor_parallel(args):
    """Each node holds a column slice of weight matrices.
    RDMA gather combines partial outputs."""

    device = torch.device("mps")
    model = DemoModel(hidden=args.hidden, depth=args.depth).to(device)
    plan = ModelParallelPlan(world_size=args.world_size)

    # Split each Linear layer's weight column-wise across nodes
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            shards = plan.tensor_parallel_split(param.data, dim=0)
            param.data = shards[args.node_id].local_data

    dist_model = DistributedModel(
        model, node_id=args.node_id, world_size=args.world_size
    )
    dist_model.register_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[node {args.node_id}] Tensor-parallel training "
          f"(hidden={args.hidden}, depth={args.depth})")

    for step in range(args.steps):
        x, y = make_batch(args.batch_size)

        optimizer.zero_grad(set_to_none=True)

        # Forward: each node computes its column shard
        # Output needs RDMA all-reduce across nodes to reconstruct full output
        output = dist_model(x)
        loss = loss_fn(output, y)
        loss.backward()

        dist_model.sync_gradients()
        optimizer.step()

        if args.node_id == 0 or step % 10 == 0:
            print(f"[node {args.node_id}] step {step}: loss={loss.item():.4f}")

    dist_model.deregister_parameters()


# ── Pipeline-parallel training ─────────────────────────────────────────────

def train_pipeline_parallel(args):
    """Layers are split across nodes in sequence.
    Micro-batches flow through the pipeline with RDMA transfers between stages."""

    device = torch.device("mps")
    full_model = DemoModel(hidden=args.hidden, depth=args.depth)
    plan = ModelParallelPlan(world_size=args.world_size)

    # Split layers across nodes
    all_layers = list(full_model.net.children())
    stage_layers = plan.pipeline_parallel_split(all_layers)[args.node_id]
    stage_model = nn.Sequential(*stage_layers).to(device)

    dist_model = DistributedModel(
        stage_model, node_id=args.node_id, world_size=args.world_size
    )
    dist_model.register_parameters()

    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    micro_batches = args.micro_batches
    per_micro = args.batch_size // micro_batches

    print(f"[node {args.node_id}] Pipeline-parallel training "
          f"(stage={args.node_id}/{args.world_size}, "
          f"micro_batches={micro_batches})")

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for mb in range(micro_batches):
            x, y = make_batch(per_micro)

            # In a full implementation, data flows:
            #   node 0 forward → RDMA → node 1 forward → RDMA → node 2 forward
            #   node 2 backward → RDMA → node 1 backward → RDMA → node 0 backward
            output = dist_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            total_loss += loss.item()

        dist_model.sync_gradients()
        optimizer.step()

        if args.node_id == 0 or step % 10 == 0:
            print(f"[node {args.node_id}] step {step}: "
                  f"loss={total_loss/micro_batches:.4f}")

    dist_model.deregister_parameters()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="distro distributed training")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument(
        "--mode", choices=["data_parallel", "tensor_parallel", "pipeline_parallel"],
        default="data_parallel",
    )
    args = parser.parse_args()

    # Initialize DMEM
    distro.init_dmem(node_id=args.node_id, num_peers=args.world_size)
    print(f"[node {args.node_id}] DMEM initialized "
          f"(world_size={args.world_size})")

    if args.mode == "data_parallel":
        train_data_parallel(args)
    elif args.mode == "tensor_parallel":
        train_tensor_parallel(args)
    elif args.mode == "pipeline_parallel":
        train_pipeline_parallel(args)

    distro.shutdown_dmem()
    print(f"[node {args.node_id}] Shutdown complete.")


if __name__ == "__main__":
    main()
