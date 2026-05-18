# Distributed Metal GPU Runtime — Architecture Plan

## Context

Transforming MCCL from a `torch.distributed` collective-communications backend into a **distributed Metal runtime** for a 3-node cluster of Mac minis connected by a full Thunderbolt mesh. The cluster presents as a **single logical GPU** with a global address space, a release-consistency coherence protocol, configurable data/model sharding, and multi-tenant scheduling.

**Key enabling property of Apple Silicon:** unified memory. An `MTLBuffer` allocated with shared storage has a CPU-addressable pointer (`buffer.contents`) that aliases the same physical pages the GPU sees. That means any byte the network stack writes to that pointer is immediately visible to a GPU shader without a copy. This is the closest analog to "GPUDirect" available on macOS — not because the NIC DMAs into GPU memory (there is no RDMA NIC on a Mac), but because there is no separate GPU memory in the first place.

**What this plan deliberately does *not* assume:**
- No ibverbs / RDMA / InfiniBand. macOS has no RDMA stack. All "one-sided" semantics in this plan are implemented over TCP sockets on the Thunderbolt-bridge subnets. We keep RDMA-style *API shapes* (put/get, rkeys, completion queues) because they're a clean model, but the wire transport is TCP.
- No kernel bypass, no MSI-X, no DPDK. macOS doesn't expose these to userspace without a DriverKit extension, which is out of scope.
- No cross-node `MTLSharedEvent`. MTLSharedEvent is intra-process / intra-machine. Cross-node ordering uses TCP byte-order + explicit barriers.
- No GPU peer-to-peer. There is exactly one GPU per Mac mini.

---

## Hardware reality

Per node (current cluster: 3× Mac mini, M4 / M4 Pro / M4 Max class):

- **3 Thunderbolt ports** on the SoC. Currently 2 are used per node to form a full triangle, each peer-pair on its own `/24`. One port per node sits idle and is the obvious next bandwidth lever.
- **Per-cable real throughput ~3 GB/s** over a single TCP socket on a TB-bridge subnet. Hardware ceiling per cable is higher (TB4 ≈ 5 GB/s usable, TB5 ≈ 10 GB/s usable); the gap is BSD-TCP overhead, single-stream HOL, and small socket buffers.
- **Striping multiple sockets across multiple TB cables to the same peer scales close to linearly** until the SoC's PCIe fabric saturates. This is the single largest perf lever in the system.
- **Unified memory bandwidth dwarfs the network** (M4 base ≈ 120 GB/s, M4 Pro ≈ 270 GB/s, M4 Max ≈ 540 GB/s). Local reductions are essentially free relative to the wire; design the system around minimizing bytes-on-wire, not around minimizing local FLOPs.
- **Tailscale `100.x` mesh** is available on every node and used for control plane (rendezvous, membership, heartbeats). Never used for data.

---

## What Stays from the existing tree

| Component | Files | Why |
|---|---|---|
| Metal interop | `csrc/metal/MPSInterop.mm` | `extract_mps_buffer()` gives the CPU pointer for zero-copy network I/O |
| Event sync (intra-node) | `csrc/metal/EventSync.mm` | MTLSharedEvent for ordering shaders against CPU-side network completion, intra-node only |
| Metal kernels | `csrc/metal/MetalKernels.mm`, `shaders.metal` | Reduction / accumulate kernels |
| Progress engine | `csrc/runtime/ProgressEngine.cpp` | Async work queue, reused for transport completions and coherence ops |
| Memory pool | `csrc/runtime/MemoryPool.cpp` | Page-aligned (16 KB on Apple Silicon) allocator |
| Rendezvous | `csrc/runtime/Rendezvous.cpp` | Store-based node discovery |
| Health / Logging / Errors | `csrc/runtime/HealthMonitor.cpp`, `csrc/common/` | |
| PeerMesh (TB-TCP transport) | `csrc/backend/PeerMesh.{hpp,cpp}` | Already does per-peer TCP over the TB-bridge `/24`s; extend to multi-socket striping |
| Build system | `setup.py` | clang++ + Metal + Accelerate, metallib compilation, pybind11 |

## What Gets Removed / Reframed

- The previous `plan.md` described an `ibverbs` / RDMA transport (`csrc/transport/rdma/`). That code path is not viable on macOS and is being repurposed: the API surface (one-sided put/get with rkeys) is kept as an internal abstraction, but the implementation lives over TCP on TB-bridge subnets, not over verbs. Files under `csrc/transport/rdma/` will be renamed to reflect that (e.g. `csrc/transport/onesided/`).
- TCP `Connection*` / generic `Protocol.hpp` and any compression stages tied to the old c10d backend are removed.
- `csrc/metal/AccelerateOps*` (vDSP) and `csrc/runtime/Watchdog*` are removed.

---

## Layer 1: Distributed Memory (DMEM)

Global address space, implemented over TCP-on-Thunderbolt. Each node "registers" a region by recording `(base_ptr, length, flags)` in a local table and assigning it a stable `region_id`. The cluster-wide catalog (`MemoryCatalog`) is a replicated directory of `(node_id, region_id, length)`; resolution to a peer's local address happens at the peer (the local table is the authoritative map). This is identical to RDMA semantics from the user's point of view; the difference is that "remote put" turns into "TCP write to the peer's daemon, which `memcpy`s into the local pointer."

**Key data structures:**
- `GlobalAddress {node_id, region_id, offset}` — 14-byte wire format
- `MemoryRegion {gaddr, local_addr, length, flags}` — registered region (no rkey/lkey on macOS; the peer's local table is the access control)
- `MemoryCatalog` — replicated directory of all cluster regions

**Key API:**
```cpp
uint32_t register_region(void* base, uint64_t len, Flags flags);
uint32_t register_mtl_buffer(id<MTLBuffer> buf, uint64_t offset, uint64_t len, Flags flags);
uint64_t put(uint16_t target, GlobalAddress dst, const void* src, uint64_t len);
uint64_t get(uint16_t target, GlobalAddress src, void* dst, uint64_t len);
uint64_t put_with_imm(uint16_t target, GlobalAddress dst, const void* src, uint64_t len, uint32_t imm);
```

**Zero-copy path:** `extract_mps_buffer()` → shared-storage `cpu_ptr` → fed directly into `send(2)` / `recv(2)`. No staging buffer between socket and GPU memory because there is no separate GPU memory. Private-storage MTLBuffers (rare in this codebase) still need a blit; avoid them.

**Multi-cable striping:** the transport opens **K sockets per peer**, one per TB `/24` between the two nodes. Bulk transfers are chunked across the K sockets. K=2 today (two TB cables per pair), expandable to K=3 once the spare `en4` ports are wired.

**Threading:** dedicated completion-poller thread per peer connection (or one `kqueue` thread fanning out to all peers — TBD by benchmarking). Pin to a P-core via QoS class `USER_INTERACTIVE`.

---

## Layer 2: Coherence Protocol

**Release consistency** on top of a **directory-based** protocol. Home node = the node that allocated the memory. 64 KB coherence granules (4× Apple Silicon page size, efficient directory overhead).

**States per granule:** UNCACHED → SHARED (readers exist) → EXCLUSIVE (one writer)

**Protocol messages** delivered as small framed TCP writes to a well-known per-node control socket:
- `READ_REQ` / `READ_REPLY` — acquire a readable copy
- `EXCLUSIVE_REQ` / `EXCLUSIVE_GRANT` — acquire write permission
- `INVALIDATE` / `INVAL_ACK` — force sharers to discard on write
- `WRITEBACK_REQ` / `WRITEBACK_ACK` — flush dirty exclusive data home

**Key API:**
```cpp
void acquire(uint64_t epoch);            // ensure all prior remote writes visible
void release();                          // flush local writes before others read
void fence();                            // full memory fence
void barrier_all();                      // cluster-wide sync point
void read_page(GlobalAddress, bool write); // coherence-managed granule fetch
void gpu_acquire(id<MTLCommandBuffer>, uint64_t epoch);  // encode MTLSharedEvent wait, intra-node
void gpu_release(id<MTLCommandBuffer>, uint64_t epoch);  // encode MTLSharedEvent signal, intra-node
```

**GPU integration:** `gpu_acquire()` / `gpu_release()` encode `encodeWaitForEvent` / `encodeSignalEvent` on the local command buffer to order shader execution against the *local* network completion. Cross-node ordering is provided by the protocol messages themselves (TCP delivers in order on each socket, and the directory mutex serializes transitions at the home node).

**Ordering:** TCP guarantees in-order delivery per socket. With K sockets per peer, ordering is preserved by binding each granule to a deterministic socket (e.g. `socket = hash(region_id, offset) mod K`). Directory lock never held across network ops (deadlock-free).

---

## Layer 3: Distributed Metal Runtime

Presents a single logical MTLDevice abstraction over N physical devices (one per node). Each node compiles the same metallib locally. The runtime coordinates which shader runs on which data shard.

**Two execution models:**
1. **Data-parallel:** each node runs the same shader on its grid partition. Before dispatch: `gpu_acquire()` on remote input buffers (after coherence has fetched them). After dispatch: `gpu_release()` on output buffers. `barrier_all()` at end.
2. **Pipeline-parallel:** nodes form a pipeline with TB-TCP transfers between stages. Double-buffered: one transfer overlaps with computation. Warm-up → steady-state → cool-down.

**Key API:**
```cpp
DistributedDevice init_device();
id<MTLLibrary> load_library(const char* metallib_path);
void dispatch_data_parallel(ShaderDispatch& dispatch, ResourceBindFn bind);
void dispatch_pipeline_parallel(vector<PipelineStage>& stages, int num_micro_batches);
```

**Note on tensor-parallel:** the network ceiling (~3 GB/s per cable, ~6 GB/s per peer with 2 cables striped) means cross-node tensor-parallel matmul is only viable for small hidden dims or with aggressive overlap. This plan supports it but does not optimize for it; DP / ZeRO / PP are the realistic regimes on this hardware.

**Threading:** dispatch calls are blocking. Transfer pool sized to K-sockets-per-peer for pipeline stage transfers.

---

## Layer 4: Tensor Runtime (Python)

User-facing API. `DistributedTensor` wraps a local tensor shard + its DMEM global address. Gradient sync is N-1 independent TCP-over-TB writes per node — no MPI-style collective state machine, just point-to-point sends fanned out.

```python
class DistributedTensor:
    def __init__(self, local_tensor, sharding, shard_index, num_shards, global_shape, dtype)
    def sync_gradient(self)      # put() to each peer's gradient buffer
    def all_reduce(self, op)     # direct point-to-point writes, no collective FSM
    def all_gather(self)         # peer get()s
    def reduce_scatter(self)     # peer get() + local accumulate

class ModelParallelPlan:
    def data_parallel_split(tensor, dim)    # chunk along batch dim
    def tensor_parallel_split(weight, dim)  # chunk along feature dim
    def pipeline_parallel_split(layers)     # assign layers to stages
```

Distributed training loop: forward → backward → `sync_gradient()` → optimizer.step().

**Wire-format optimization:** dtype policy allows bf16-on-the-wire even when local accumulators are fp32. M4 GPUs do bf16 natively; the cast is a single trivial shader. Halves bytes-on-wire for gradient sync.

---

## Layer 5: Cluster Manager

Daemon per node. **Tailscale `100.x` TCP** for control channel (membership, heartbeats, catalog sync, job submission). **TB-bridge TCP** for data only.

**Components:**
- **Membership:** leader election (lowest node_id), 500 ms heartbeats over Tailscale, 3-miss detection (~1.5 s)
- **Region registry:** replicated catalog of all DMEM regions
- **Scheduler** (leader): priority queue + fair-share, 1 s tick interval. Supports preemption.
- **Fault tolerance:** on node failure, catalog rebuild + job rescheduling on remaining nodes

**Key API:**
```cpp
ClusterState get_cluster_state();
uint64_t submit_job(JobDescriptor);
void schedule_tick();
void handle_node_failure(uint16_t node_id);
```

**Threads:** control listener (Tailscale fd), heartbeat sender, scheduler tick (leader only).

---

## New File Structure

```
csrc/
├── dmem/
│   ├── DistributedMemoryManager.{hpp,cpp}
│   ├── GlobalAddress.hpp
│   └── MemoryCatalog.hpp
├── coherence/
│   ├── CoherenceProtocol.{hpp,cpp}
│   ├── CoherenceDirectory.hpp
│   └── CoherenceMessage.hpp
├── runtime/
│   ├── DistributedMetalRuntime.{hpp,cpp}
│   ├── ShaderDispatch.hpp
│   ├── ClusterManager.{hpp,cpp}
│   ├── Scheduler.{hpp,cpp}
│   ├── ProgressEngine.{hpp,cpp}     (survives)
│   ├── Rendezvous.{hpp,cpp}         (survives)
│   ├── HealthMonitor.{hpp,cpp}      (survives)
│   ├── Metrics.{hpp,cpp}            (survives)
│   └── MemoryPool.{hpp,cpp}         (survives)
├── python/
│   └── bindings.cpp                 (pybind11)
├── transport/onesided/              (formerly transport/rdma/, rewritten as TCP-over-TB)
│   ├── PeerLink.{hpp,cpp}           (K sockets per peer, striping)
│   ├── SharedBuffer.{hpp,cpp}
│   └── Framing.{hpp,cpp}
├── metal/
│   ├── MPSInterop.{hpp,mm}
│   ├── EventSync.{hpp,mm}           (intra-node only)
│   ├── MetalKernels.{hpp,mm}
│   └── shaders.metal
└── common/
    ├── Errors.hpp
    ├── Logging.hpp
    └── Version.hpp
mccl/
├── __init__.py
├── config.py
├── version.py
├── tuning.py
└── distributed/
    ├── __init__.py
    ├── tensor.py                    # DistributedTensor
    ├── sharding.py                  # ModelParallelPlan
    └── cluster.py                   # Python ClusterManager client
```

---

## Implementation Sequence

**Phase 0 — Transport foundation:** PeerLink with K sockets per peer over the existing TB `/24`s; chunked striped send/recv; P-core QoS for the I/O thread; large SO_SNDBUF/RCVBUF; TCP_NODELAY. Bench point-to-point throughput; verify ≥ 5 GB/s with K=2.

**Phase 1 — DMEM:** DistributedMemoryManager (register, put, get, catalog) on top of PeerLink → completion-poller thread → two-node 1 GB cross-put / cross-get test.

**Phase 2 — Coherence:** CoherenceDirectory + state tracking → message protocol → acquire/release/fence/barrier_all → MTLSharedEvent integration (intra-node) → coherence stress test.

**Phase 3 — Distributed Runtime:** ShaderDispatch + dispatch_data_parallel → tensor transfer via DMEM → dispatch_pipeline_parallel with micro-batching → correctness test (distributed matmul across 3 nodes).

**Phase 4 — Tensor Runtime:** DistributedTensor + DMEM registration → sync_gradient via put() → all_reduce / all_gather / reduce_scatter → bf16-on-the-wire policy → ModelParallelPlan → convergence test (training loop).

**Phase 5 — Cluster Manager:** Tailscale control channel + heartbeats → membership + leader election → job scheduler → fault tolerance (kill a node, verify recovery).

---

## Verification

1. **Transport:** Two nodes, K=2 sockets per peer over two TB cables, sustain ≥ 5 GB/s on a 10 GB transfer.
2. **DMEM:** Two nodes, register 1 GB MTLBuffer on each, cross-put and cross-get, verify data integrity with checksums.
3. **Coherence:** Three nodes, concurrent read/write patterns to same granule, verify directory state machine correctness via assertions on epoch ordering.
4. **Distributed Runtime:** Data-parallel matrix multiply across 3 nodes, compare output to single-node reference (bit-exact for fp32).
5. **Tensor Runtime:** Full training loop with a known model on known data, verify loss curve matches single-node reference within numerical tolerance.
6. **Cluster Manager:** Kill node 2 mid-job, verify job continues on nodes 0 and 1 with correct gradient sync for 2-way.
7. **End-to-end:** `torchrun --nnodes=3 --nproc_per_node=1 train.py` with distributed sharding, measure throughput scaling vs single node.
