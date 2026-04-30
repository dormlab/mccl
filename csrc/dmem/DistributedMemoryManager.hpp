#pragma once

#include "dmem/GlobalAddress.hpp"
#include "dmem/MemoryCatalog.hpp"
#include "transport/rdma/RdmaConnection.hpp"
#include "transport/rdma/SharedBuffer.hpp"
#include "transport/rdma/IbvWrapper.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <deque>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <future>

namespace distro {

static constexpr int MAX_NODES = 8;

/// Distributed Memory Manager — the PGAS layer.
///
/// Each node registers its GPU buffers (MTLBuffer-backed) with the local RDMA
/// NIC via ibv_reg_mr, then advertises the resulting (addr, length, rkey) to
/// all peers through the MemoryCatalog.  Other nodes resolve a GlobalAddress
/// into RDMA credentials and issue one-sided put()/get() operations.
///
/// On Apple Silicon, unified memory means MTLBuffer.contents is a CPU-visible
/// pointer — registering it with the RDMA NIC gives the NIC direct DMA access
/// to GPU memory.  No bounce buffers, no driver hacks.
///
/// Thread safety: all public methods are thread-safe.  RDMA post operations
/// are serialized per QP — each peer's RdmaConnection is owned by the DMEM
/// poller thread at submission time.
class DistributedMemoryManager {
public:
    /// Configuration for the DMEM subsystem.
    struct Config {
        uint16_t node_id = 0;
        int      num_peers = 0;
        int      cq_depth = 256;
    };

    DistributedMemoryManager(const Config& config);
    ~DistributedMemoryManager();

    DistributedMemoryManager(const DistributedMemoryManager&) = delete;
    DistributedMemoryManager& operator=(const DistributedMemoryManager&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────

    /// Start the completion poller thread.  Must be called after
    /// RdmaConnection objects have been opened for all peers.
    void start();

    /// Drain pending ops and stop the poller thread.
    void shutdown();

    // ── Region management ────────────────────────────────────────────

    /// Register a buffer (e.g., MTLBuffer.contents pointer) with the local
    /// RDMA NIC and publish it to the cluster catalog.
    /// Returns the assigned region_id (>= 1).
    uint32_t register_region(void* base_addr, uint64_t length,
                             uint8_t flags);

    /// Unregister and remove from the catalog.
    void unregister_region(uint32_t region_id);

    /// Register an MTLBuffer-backed tensor for RDMA access.
    /// `cpu_ptr` comes from MPSInterop::extract_mps_buffer().cpu_ptr.
    uint32_t register_buffer(void* cpu_ptr, uint64_t length,
                             uint8_t flags);

    // ── Address resolution ───────────────────────────────────────────

    /// Resolve a GlobalAddress to local or remote access parameters.
    AddrResolution resolve(GlobalAddress addr) const;

    /// Look up a region's RDMA credentials for a remote node.
    bool lookup_remote(uint16_t node_id, uint32_t region_id,
                       uint64_t& remote_addr, uint32_t& rkey,
                       uint64_t& length) const;

    // ── One-sided RDMA operations ────────────────────────────────────

    /// RDMA write: put local data into remote memory.
    /// Non-blocking: returns a wr_id for completion polling via poll_completion().
    uint64_t put(uint16_t target_node,
                 GlobalAddress dst_addr,
                 const void* src_local, uint64_t length);

    /// RDMA read: pull remote data into local memory.
    /// Non-blocking: returns a wr_id for completion polling.
    uint64_t get(uint16_t target_node,
                 GlobalAddress src_addr,
                 void* dst_local, uint64_t length);

    /// RDMA write with immediate data (used for coherence protocol messages).
    /// The imm_data is delivered to the remote CQ on completion.
    uint64_t put_with_imm(uint16_t target_node,
                          GlobalAddress dst_addr,
                          const void* src_local, uint64_t length,
                          uint32_t imm_data);

    /// RDMA fence on a peer's QP.  Ensures all prior RDMA ops to this
    /// peer complete before subsequent ones begin.
    uint64_t fence(uint16_t target_node);

    // ── Completion tracking ──────────────────────────────────────────

    /// Block until the RDMA op identified by wr_id completes.
    /// Returns true on success, false on timeout or error.
    bool poll_completion(uint64_t wr_id,
                         std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    /// Wait for all pending operations to complete.
    void drain_pending();

    // ── Peer connection access ───────────────────────────────────────

    /// Access a peer's RDMA connection (for direct QP manipulation).
    RdmaConnection* peer_connection(uint16_t node_id);

    /// Protection domain shared across all QPs on this node.
    ibv_pd* pd() const { return pd_; }

    /// Local node ID.
    uint16_t node_id() const { return node_id_; }

    // ── Statistics ───────────────────────────────────────────────────

    struct Stats {
        uint64_t total_put_bytes = 0;
        uint64_t total_get_bytes = 0;
        uint64_t total_put_ops   = 0;
        uint64_t total_get_ops   = 0;
        uint64_t total_errors    = 0;
    };
    Stats stats() const;

private:
    static constexpr size_t kRdmaBufSize = 1024 * 1024;  // 1 MB max chunk
    static constexpr int    kCqPollBatch = 16;

    // ── Completion poller ────────────────────────────────────────────
    void poller_loop();
    void process_completions();

    // ── Chunked RDMA helpers ─────────────────────────────────────────
    uint64_t put_chunked(uint16_t target, uint64_t remote_addr, uint32_t rkey,
                         const void* src, uint64_t length);
    uint64_t get_chunked(uint16_t target, uint64_t remote_addr, uint32_t rkey,
                         void* dst, uint64_t length);

    // ── Member variables ─────────────────────────────────────────────
    uint16_t node_id_;
    int      num_peers_;

    // RDMA device infrastructure (shared across all QPs)
    const IbvFunctions* ibv_ = nullptr;
    ibv_context*        ctx_ = nullptr;
    ibv_pd*             pd_  = nullptr;
    ibv_cq*             cq_  = nullptr;
    ibv_device*         dev_ = nullptr;

    // Per-peer connections (indexed by node_id, self entry is unused)
    std::vector<std::unique_ptr<RdmaConnection>> peer_conns_;

    // Local memory regions
    std::vector<MemoryRegion> regions_;
    mutable std::mutex regions_mu_;
    std::atomic<uint32_t> next_region_id_{1};  // 0 = catalog

    // Memory catalog
    MemoryCatalog catalog_;
    SharedBuffer catalog_buf_;
    ibv_mr* catalog_mr_ = nullptr;

    // Staging buffer for chunked operations
    SharedBuffer staging_buf_;
    ibv_mr* staging_mr_ = nullptr;

    // Completion tracking
    struct PendingOp {
        uint64_t wr_id;
        std::promise<void> done;
    };
    std::deque<PendingOp> pending_ops_;
    std::mutex pending_mu_;
    std::condition_variable pending_cv_;

    // Poller thread
    std::thread poller_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_wr_id_{1};

    // Per-peer submission mutex (one QP = one submitter at a time)
    std::vector<std::unique_ptr<std::mutex>> peer_mu_;

    // Stats
    mutable std::mutex stats_mu_;
    Stats stats_;
};

} // namespace distro
