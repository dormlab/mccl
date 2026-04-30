#pragma once

#include "coherence/CoherenceMessage.hpp"
#include "coherence/CoherenceDirectory.hpp"
#include "dmem/DistributedMemoryManager.hpp"

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <functional>

namespace distro {

/// Release-consistency coherence protocol with directory at the head node.
///
/// The coherence protocol ensures that RDMA reads and writes across the
/// cluster observe a consistent view of memory.  It uses:
///   - Directory-based tracking at the home node (head)
///   - 64 KB page granules
///   - MTLSharedEvent integration for GPU-side acquire/release
///   - RDMA write_with_imm for protocol messages
///
/// Programming model (matching GPU weak memory):
///   acquire() — ensure all prior remote writes are visible
///   release() — flush local writes before others read
///   fence()   — full memory fence on this node
///   barrier_all() — cluster-wide synchronization
///
/// The head node runs the full directory; agents run a minimal client
/// that handles invalidations and writebacks on command.

class CoherenceProtocol {
public:
    struct Config {
        uint16_t node_id = 0;
        bool     is_head = false;  // True on the head node (runs directory)
        int      num_nodes = 3;
    };

    CoherenceProtocol(const Config& config, DistributedMemoryManager* dmem);
    ~CoherenceProtocol();

    CoherenceProtocol(const CoherenceProtocol&) = delete;
    CoherenceProtocol& operator=(const CoherenceProtocol&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────

    void start();
    void shutdown();

    // ── Synchronization primitives ───────────────────────────────────

    /// Acquire: ensure all prior remote writes to this node are visible.
    /// Blocks until pending RDMA reads complete and coherence state
    /// reflects the latest epoch from the home node.
    void acquire();

    /// Release: flush all locally-modified exclusive pages back to the
    /// home node so that subsequent acquires by other nodes see them.
    void release();

    /// Full fence: all prior coherence operations complete before
    /// subsequent ones begin.  Maps to RDMA fence + local sync.
    void fence();

    /// Cluster-wide barrier.  All nodes must call before any proceeds.
    void barrier_all();

    // ── GPU-side synchronization (MTLSharedEvent) ────────────────────

    /// Encode a wait on the command buffer: shader waits until all
    /// coherence data for the given epoch is visible in local memory.
    /// Called before a shader reads remote data.
    void gpu_acquire(void* cmd_buffer, uint64_t epoch);

    /// Encode a signal on the command buffer: after the shader finishes
    /// writing, the coherence engine will start writeback.
    /// Called after a shader writes to remote data.
    void gpu_release(void* cmd_buffer, uint64_t epoch);

    // ── Page-level operations ────────────────────────────────────────

    /// Read a remote page into local memory with coherence tracking.
    /// If write=true, requests exclusive access for subsequent writes.
    void read_page(GlobalAddress page, bool write = false);

    /// Mark a locally-held page as dirty (after writing).
    void write_page_done(GlobalAddress page);

    // ── Directory operations (head only) ─────────────────────────────

    /// Handle an incoming coherence message from a peer.
    void handle_message(const CoherenceMessage& msg, uint16_t sender);

    /// Invalidate all sharers of a page (called when a writer requests
    /// exclusive access).  Sends INVALIDATE to each sharer.
    void invalidate_sharers(uint32_t region_id, uint64_t page_idx);

    // ── Health ───────────────────────────────────────────────────────

    /// Current global epoch (monotonically increasing).
    uint64_t global_epoch() const { return global_epoch_.load(); }

private:
    // ── Message delivery ─────────────────────────────────────────────
    void send_message(uint16_t target, const CoherenceMessage& msg);
    void send_message_with_data(uint16_t target, CoherenceOp op,
                                 GlobalAddress page,
                                 uint64_t epoch);

    // ── Local cache tracking ─────────────────────────────────────────
    struct CachedPage {
        GlobalAddress gaddr;
        CoherenceState local_state;
        uint64_t epoch;
        bool dirty;
    };
    CachedPage* find_cached(GlobalAddress page);
    void track_page(GlobalAddress page, CoherenceState state, uint64_t epoch);
    void untrack_page(GlobalAddress page);

    // ── Member variables ─────────────────────────────────────────────
    uint16_t node_id_;
    bool     is_head_;
    int      num_nodes_;

    DistributedMemoryManager* dmem_;

    // Coherence message buffers (one per peer, RDMA-registered)
    struct MsgBuf {
        std::unique_ptr<uint8_t[]> buf;
        uint64_t local_addr;
        uint32_t region_id;
        uint32_t rkey;
        uint32_t lkey;
    };
    std::unordered_map<uint16_t, MsgBuf> msg_bufs_;

    // Directory (only populated on head)
    CoherenceDirectory directory_;

    // Local page cache
    std::unordered_map<uint64_t, CachedPage> local_cache_;
    std::mutex cache_mu_;

    // Global epoch
    std::atomic<uint64_t> global_epoch_{0};

    // Poller thread
    std::thread poller_thread_;
    std::atomic<bool> running_{false};

    // Barrier counter
    std::atomic<uint32_t> barrier_count_{0};
};

} // namespace distro
