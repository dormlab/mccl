# Distributed Metal GPU Runtime — Architecture Plan

## Context

Transforming MCCL from a `torch.distributed` collective-communications backend into a **distributed Metal runtime** for a 3-node cluster of Mac minis (Thunderbolt 5 RDMA fabric). The cluster presents as a **single logical GPU** with full PGAS, a release-consistency coherence protocol, configurable data/model sharding, and multi-tenant scheduling. ~80-85% of the current codebase is removed; the RDMA transport, Metal interop, event sync, and progress engine survive.

**Key enabling insight:** Apple Silicon unified memory means `MTLBuffer.contents` is a valid CPU pointer. Registering that pointer with the RDMA NIC (`ibv_reg_mr`) gives the NIC direct DMA access to GPU memory — no bounce buffers, no driver hacks. This is GPUDirect RDMA on every Apple Silicon Mac.

---

## What Stays (~15-20%)

| Component | Files | Why |
|---|---|---|
| RDMA transport | `csrc/transport/rdma/{RdmaConnection,IbvWrapper,SharedBuffer}.*`, `ibverbs_compat.h` | Extended with one-sided RDMA READ/WRITE ops |
| Metal interop | `csrc/metal/MPSInterop.mm` | `extract_mps_buffer()` gives us the CPU pointer for zero-copy RDMA registration |
| Event sync | `csrc/metal/EventSync.mm` | MTLSharedEvent protocol extended for cross-node coherence |
| Metal kernels | `csrc/metal/MetalKernels.mm`, `shaders.metal` | Reduction/accumulate kernels, stripped of collectives cruft |
| Progress engine | `csrc/runtime/ProgressEngine.cpp` | Async work queue pattern reused for DMEM/coherence ops |
| Memory pool | `csrc/runtime/MemoryPool.cpp` | Page-aligned allocator extended for distributed regions |
| Rendezvous | `csrc/runtime/Rendezvous.cpp` | Store-based node discovery extended for cluster membership |
| Health/Logging/Errors | `csrc/runtime/HealthMonitor.cpp`, `csrc/common/` | Peer liveness, logging macros, exception hierarchy |
| Build system | `setup.py` patterns | clang++ with Metal/Accelerate, metallib compilation, pybind11 added |

## What Gets Removed (~80-85%)

`csrc/backend/` (ProcessGroupMCCL, WorkMCCL, MPSDispatch, Registration), `csrc/transport/TcpTransport*`, `csrc/transport/Connection*`, `csrc/transport/Protocol.hpp`, `csrc/compression/`, `csrc/runtime/Watchdog*`, `csrc/metal/AccelerateOps*` (vDSP)

---

## Layer 1: Distributed Memory (DMEM)

Global address space: each node registers its Metal buffers with the local RDMA NIC, then advertises `(node_id, region_id, base_addr, length, rkey)` to all peers. Other nodes resolve a `GlobalAddress` → RDMA remote addr + rkey → one-sided `put()`/`get()`.

**Key data structures:**
- `GlobalAddress {node_id, region_id, offset}` — 14-byte wire format
- `MemoryRegion {gaddr, local_addr, length, rkey, lkey, flags}` — registered RDMA region
- `MemoryCatalog` — replicated directory of all cluster regions, each node exposes its catalog at a well-known RDMA-addressable buffer

**Key API:**
```cpp
uint32_t register_region(void* base, uint64_t len, Flags flags);
uint32_t register_mtl_buffer(id<MTLBuffer> buf, uint64_t offset, uint64_t len, Flags flags);
uint64_t put(uint16_t target, GlobalAddress dst, const void* src, uint64_t len);
uint64_t get(uint16_t target, GlobalAddress src, void* dst, uint64_t len);
uint64_t put_with_imm(uint16_t target, GlobalAddress dst, const void* src, uint64_t len, uint32_t imm);
```

**RdmaConnection extensions** (added to existing class):
```cpp
bool post_rdma_write(ibv_sge& sge, uint64_t remote_addr, uint32_t rkey, uint64_t wr_id);
bool post_rdma_read(ibv_sge& sge, uint64_t remote_addr, uint32_t rkey, uint64_t wr_id);
bool post_rdma_write_with_imm(ibv_sge& sge, uint64_t remote_addr, uint32_t rkey, uint32_t imm, uint64_t wr_id);
```

**Zero-copy path:** `extract_mps_buffer()` → `cpu_ptr` (shared storage) → `ibv_reg_mr(pd, cpu_ptr, nbytes, LOCAL_WRITE|REMOTE_READ|REMOTE_WRITE)` → NIC DMA directly to/from GPU memory. Private storage still needs a staging blit.

**Threading:** Dedicated CQ poller thread dispatches completions. One QP per peer, RC transport.

---

## Layer 2: Coherence Protocol

**Release consistency** on top of a **directory-based** protocol. Home node = the node that allocated the memory. 64KB coherence granules (16 RDMA MTUs, efficient directory overhead).

**States per page:** UNCACHED → SHARED (readers exist) → EXCLUSIVE (one writer)

**Protocol messages** delivered via `put_with_imm()` to well-known per-node message buffers:
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
void read_page(GlobalAddress, bool write); // coherence-managed page fetch
void gpu_acquire(id<MTLCommandBuffer>, uint64_t epoch);  // encode MTLSharedEvent wait
void gpu_release(id<MTLCommandBuffer>, uint64_t epoch);  // encode MTLSharedEvent signal
```

**GPU integration:** `gpu_acquire()` encodes `encodeWaitForEvent` on the command buffer before shader execution. `gpu_release()` encodes `encodeSignalEvent` after writes. Coherence poller thread monitors `signaledValue`, triggers writeback for dirty exclusive pages.

**Ordering:** RDMA RC guarantees prior ops complete before subsequent ones on the same QP. Directory mutex serializes all transitions at the home node. Directory lock never held across network ops (deadlock-free).

---

## Layer 3: Distributed Metal Runtime

Presents a single logical MTLDevice. Each node compiles the same metallib locally. The runtime coordinates which shader runs on which data shard.

**Two execution models:**
1. **Data-parallel:** each node runs the same shader on its grid partition. Before dispatch: `gpu_acquire()` on remote input buffers. After dispatch: `gpu_release()` on output buffers. `barrier_all()` at end.
2. **Pipeline-parallel:** nodes form a pipeline with RDMA transfers between stages. Double-buffered: one transfer overlaps with computation. Warm-up → steady-state → cool-down.

**Key API:**
```cpp
DistributedDevice init_device();
id<MTLLibrary> load_library(const char* metallib_path);
void dispatch_data_parallel(ShaderDispatch& dispatch, ResourceBindFn bind);
void dispatch_pipeline_parallel(vector<PipelineStage>& stages, int num_micro_batches);
```

**Threading:** Dispatch calls are blocking. Transfer pool (2 threads) handles RDMA between pipeline stages.

---

## Layer 4: Tensor Runtime (Python)

User-facing API. `DistributedTensor` wraps a local tensor shard + its DMEM global address. No collectives — gradient sync is N-1 independent RDMA writes per node.

```python
class DistributedTensor:
    def __init__(self, local_tensor, sharding, shard_index, num_shards, global_shape, dtype)
    def sync_gradient(self)      # RDMA put() to each peer's gradient buffer
    def all_reduce(self, op)     # direct RDMA writes, no MPI collectives
    def all_gather(self)         # RDMA read from every peer
    def reduce_scatter(self)     # RDMA read + local accumulate

class ModelParallelPlan:
    def data_parallel_split(tensor, dim)    # chunk along batch dim
    def tensor_parallel_split(weight, dim)  # chunk along feature dim
    def pipeline_parallel_split(layers)     # assign layers to stages
```

Distributed training loop: forward → backward → `sync_gradient()` (RDMA writes) → optimizer.step(). No allreduce collective.

---

## Layer 5: Cluster Manager

Daemon per node. TCP control channel for membership, heartbeats, catalog sync, job submission. RDMA for data.

**Components:**
- **Membership:** leader election (lowest node_id), 500ms heartbeats, 3-miss detection (~1.5s)
- **Region registry:** replicated catalog of all DMEM regions
- **Scheduler** (leader): priority queue + fair-share, 1s tick interval. Supports preemption.
- **Fault tolerance:** on node failure, catalog rebuild + job rescheduling on remaining nodes

**Key API:**
```cpp
ClusterState get_cluster_state();
uint64_t submit_job(JobDescriptor);
void schedule_tick();
void handle_node_failure(uint16_t node_id);
```

**Threads:** control listener, heartbeat sender, scheduler tick (leader only).

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
│   └── bindings.cpp                 (pybind11, replaces old Registration.cpp)
├── transport/rdma/                  (survives, extended)
│   ├── RdmaConnection.{hpp,cpp}     (extended: one-sided ops)
│   ├── RdmaTransport.{hpp,cpp}
│   ├── SharedBuffer.{hpp,cpp}
│   ├── IbvWrapper.{hpp,cpp}
│   └── ibverbs_compat.h
├── metal/                           (survives, stripped)
│   ├── MPSInterop.{hpp,mm}
│   ├── EventSync.{hpp,mm}           (extended: cross-node coherence)
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

**Phase 1 — DMEM (foundation):** Extend RdmaConnection with one-sided ops → DistributedMemoryManager (register, put, get, catalog) → CQ poller thread → two-node 1GB RDMA READ/WRITE test.

**Phase 2 — Coherence:** CoherenceDirectory + state tracking → message protocol via put_with_imm → acquire/release/fence/barrier_all → MTLSharedEvent integration → coherence stress test.

**Phase 3 — Distributed Runtime:** ShaderDispatch + dispatch_data_parallel → RDMA tensor transfer → dispatch_pipeline_parallel with micro-batching → correctness test (distributed matmul across 3 nodes).

**Phase 4 — Tensor Runtime:** DistributedTensor + DMEM registration → sync_gradient via put() → all_reduce/all_gather/reduce_scatter via DMEM → ModelParallelPlan → convergence test (training loop).

**Phase 5 — Cluster Manager:** TCP control channel + heartbeats → membership + leader election → job scheduler → fault tolerance (kill a node, verify recovery).

---

## Verification

1. **DMEM:** Two nodes, register 1GB MTLBuffer on each, cross-put and cross-get, verify data integrity with checksums
2. **Coherence:** Three nodes, concurrent read/write patterns to same page, verify directory state machine correctness via assertions on epoch ordering
3. **Distributed Runtime:** Data-parallel matrix multiply across 3 nodes, compare output to single-node reference (bit-exact for fp32)
4. **Tensor Runtime:** Full training loop with a known model on known data, verify loss curve matches single-node reference within numerical tolerance
5. **Cluster Manager:** Kill node 2 mid-job, verify job continues on nodes 0 and 1 with correct gradient sync for 2-way
6. **End-to-end:** `torchrun --nnodes=3 --nproc_per_node=1 train.py` with distributed sharding, measure throughput scaling vs single node
