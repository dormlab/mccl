#include "coherence/CoherenceProtocol.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <cstring>
#include <algorithm>

namespace distro {

// ── Construction / destruction ────────────────────────────────────────────

CoherenceProtocol::CoherenceProtocol(const Config& config,
                                       DistributedMemoryManager* dmem)
    : node_id_(config.node_id)
    , is_head_(config.is_head)
    , num_nodes_(config.num_nodes)
    , dmem_(dmem)
{
    DISTRO_CHECK(dmem_ != nullptr, "Coherence: DMEM is required");

    DISTRO_INFO("Coherence: node %u (%s) initialized",
                node_id_, is_head_ ? "head" : "agent");
}

CoherenceProtocol::~CoherenceProtocol() {
    shutdown();
}

void CoherenceProtocol::start() {
    if (running_.load()) return;
    running_ = true;

    // Allocate and register message buffers for each peer
    for (int i = 0; i < num_nodes_; i++) {
        if (i == node_id_) continue;

        MsgBuf mb;
        mb.buf = std::make_unique<uint8_t[]>(COHERENCE_MSG_BUF_SIZE);
        mb.local_addr = reinterpret_cast<uint64_t>(mb.buf.get());

        // Register with RDMA so peers can write messages to us
        uint32_t region_id = dmem_->register_region(
            mb.buf.get(), COHERENCE_MSG_BUF_SIZE,
            MemoryRegion::READABLE | MemoryRegion::WRITABLE);

        mb.region_id = region_id;

        // We'll resolve the rkey from the catalog after registration
        auto res = dmem_->resolve(GlobalAddress{node_id_, region_id, 0});
        mb.lkey = res.local_lkey;

        msg_bufs_[i] = std::move(mb);

        DISTRO_DEBUG("Coherence: msg buffer for node %u (region=%u, addr=%llx)",
                     i, region_id, (unsigned long long)mb.local_addr);
    }

    // Register the local coherence directory region (head only)
    if (is_head_) {
        // The head's message buffer region is the canonical directory
        // Agents read from it to check coherence state
    }

    DISTRO_INFO("Coherence: started (head=%s)", is_head_ ? "yes" : "no");
}

void CoherenceProtocol::shutdown() {
    if (!running_.load()) return;
    running_ = false;
    if (poller_thread_.joinable()) poller_thread_.join();
    DISTRO_INFO("Coherence: shutdown complete");
}

// ── Synchronization primitives ───────────────────────────────────────────

void CoherenceProtocol::acquire() {
    // Drain all pending RDMA reads — ensures remote data is locally visible
    dmem_->drain_pending();

    // Increment epoch to mark this acquire point
    global_epoch_.fetch_add(1);
}

void CoherenceProtocol::release() {
    // Flush all dirty exclusive pages back to the home node
    std::lock_guard<std::mutex> lock(cache_mu_);

    for (auto& [key, page] : local_cache_) {
        if (page.dirty && page.local_state == CoherenceState::EXCLUSIVE) {
            // RDMA write the dirty page data back to home
            dmem_->put(page.gaddr.node_id, page.gaddr,
                       reinterpret_cast<const void*>(
                           dmem_->resolve(page.gaddr).local_ptr),
                       COHERENCE_PAGE_SIZE);

            // Send WRITEBACK_DATA message to home
            CoherenceMessage msg{};
            msg.op        = CoherenceOp::WRITEBACK_DATA;
            msg.page_addr = page.gaddr;
            msg.epoch     = global_epoch_.fetch_add(1);

            send_message(page.gaddr.node_id, msg);

            page.dirty = false;
            page.local_state = CoherenceState::SHARED;
        }
    }

    // Drain to ensure all put() operations complete
    dmem_->drain_pending();
    global_epoch_.fetch_add(1);
}

void CoherenceProtocol::fence() {
    // RDMA fence on all peer QPs
    for (int i = 0; i < num_nodes_; i++) {
        if (i == node_id_) continue;
        dmem_->fence(i);
    }
    dmem_->drain_pending();
    global_epoch_.fetch_add(1);
}

void CoherenceProtocol::barrier_all() {
    // Simple centralized barrier using RDMA write to head's barrier region
    if (is_head_) {
        // Wait for all agents to enter
        while (barrier_count_.load() < static_cast<uint32_t>(num_nodes_ - 1)) {
            std::this_thread::yield();
        }

        // Broadcast BARRIER_LEAVE to all agents
        CoherenceMessage msg{};
        msg.op    = CoherenceOp::BARRIER_LEAVE;
        msg.epoch = global_epoch_.fetch_add(1);

        for (int i = 0; i < num_nodes_; i++) {
            if (i == node_id_) continue;
            send_message(i, msg);
        }

        barrier_count_ = 0;
    } else {
        // Agent: send BARRIER_ENTER to head, wait for BARRIER_LEAVE
        CoherenceMessage msg{};
        msg.op    = CoherenceOp::BARRIER_ENTER;
        msg.epoch = global_epoch_.fetch_add(1);

        // Find head (node 0)
        send_message(0, msg);

        // Poll for BARRIER_LEAVE — handled by handle_message()
        // Spin-wait until our epoch advances past the barrier
        uint64_t wait_epoch = msg.epoch;
        while (global_epoch_.load() <= wait_epoch + 1) {
            std::this_thread::yield();
        }
    }
}

// ── GPU-side synchronization ─────────────────────────────────────────────

void CoherenceProtocol::gpu_acquire(void* cmd_buffer, uint64_t epoch) {
    // On Apple Silicon, MTLSharedEvent is used for GPU/CPU ordering.
    // The head integrates with EventSync.mm to encode wait commands.
    // This stub is filled in when EventSync is extended for coherence.
    (void)cmd_buffer;
    (void)epoch;
}

void CoherenceProtocol::gpu_release(void* cmd_buffer, uint64_t epoch) {
    (void)cmd_buffer;
    (void)epoch;
}

// ── Page-level operations ────────────────────────────────────────────────

void CoherenceProtocol::read_page(GlobalAddress page, bool write) {
    // Align to page boundary
    page.offset = page_align_down(page.offset);

    // Check local cache first
    {
        std::lock_guard<std::mutex> lock(cache_mu_);
        CachedPage* cached = find_cached(page);
        if (cached) {
            if (write && cached->local_state == CoherenceState::SHARED) {
                // Need to upgrade: request exclusive from home
                // Falls through to the request below
            } else if (!write || cached->local_state == CoherenceState::EXCLUSIVE) {
                // Already have the right permissions
                return;
            }
        }
    }

    // Request from home node
    uint16_t home = page.node_id;

    CoherenceMessage msg{};
    msg.op        = write ? CoherenceOp::EXCLUSIVE_REQ : CoherenceOp::READ_REQ;
    msg.page_addr = page;
    msg.epoch     = global_epoch_.load();
    msg.sender_seq = 0;  // Filled by send_message

    send_message(home, msg);

    // The data arrives via RDMA write from home (or exclusive owner)
    // handle_message() processes the READ_REPLY and updates local_cache_
}

void CoherenceProtocol::write_page_done(GlobalAddress page) {
    page.offset = page_align_down(page.offset);

    std::lock_guard<std::mutex> lock(cache_mu_);
    CachedPage* cached = find_cached(page);
    if (cached && cached->local_state == CoherenceState::EXCLUSIVE) {
        cached->dirty = true;
    }
}

// ── Directory operations (head only) ─────────────────────────────────────

void CoherenceProtocol::handle_message(const CoherenceMessage& msg,
                                        uint16_t sender) {
    switch (msg.op) {
        case CoherenceOp::READ_REQ: {
            // A reader wants data for this page
            uint32_t region_id = msg.page_addr.region_id;
            uint64_t page_idx  = page_index(msg.page_addr.offset);

            DirectoryEntry& entry = directory_.get_or_create(region_id, page_idx);

            if (entry.state == CoherenceState::EXCLUSIVE) {
                // Need to fetch dirty data from exclusive owner first
                uint16_t owner = entry.exclusive_owner;
                CoherenceMessage wb_req{};
                wb_req.op        = CoherenceOp::WRITEBACK_REQ;
                wb_req.page_addr = msg.page_addr;
                wb_req.epoch     = msg.epoch;
                send_message(owner, wb_req);
                // WRITEBACK_DATA will arrive, then we respond READ_REPLY
                // (simplified: we respond READ_REPLY_DIRTY immediately)
            }

            // Add sender to sharers
            entry.add_sharer(sender);
            if (entry.state == CoherenceState::UNCACHED) {
                entry.state = CoherenceState::SHARED;
            }
            entry.epoch++;

            // Reply with data
            CoherenceMessage reply{};
            reply.op        = entry.state == CoherenceState::EXCLUSIVE
                                ? CoherenceOp::READ_REPLY_DIRTY
                                : CoherenceOp::READ_REPLY;
            reply.page_addr = msg.page_addr;
            reply.epoch     = entry.epoch;
            send_message(sender, reply);
            break;
        }

        case CoherenceOp::EXCLUSIVE_REQ: {
            uint32_t region_id = msg.page_addr.region_id;
            uint64_t page_idx  = page_index(msg.page_addr.offset);

            DirectoryEntry& entry = directory_.get_or_create(region_id, page_idx);

            if (entry.sharer_count() > 0) {
                // Need to invalidate all current sharers
                invalidate_sharers(region_id, page_idx);
            }

            if (entry.state == CoherenceState::EXCLUSIVE &&
                entry.exclusive_owner != sender) {
                // Ask current owner to write back
                CoherenceMessage wb_req{};
                wb_req.op        = CoherenceOp::WRITEBACK_REQ;
                wb_req.page_addr = msg.page_addr;
                wb_req.epoch     = msg.epoch;
                send_message(entry.exclusive_owner, wb_req);
            }

            // Grant exclusive access to the requester
            entry.state = CoherenceState::EXCLUSIVE;
            entry.exclusive_owner = sender;
            entry.clear_all_sharers();
            entry.add_sharer(sender);
            entry.epoch++;

            CoherenceMessage grant{};
            grant.op        = CoherenceOp::EXCLUSIVE_GRANT;
            grant.page_addr = msg.page_addr;
            grant.epoch     = entry.epoch;
            send_message(sender, grant);
            break;
        }

        case CoherenceOp::WRITEBACK_DATA: {
            // Exclusive owner wrote dirty data back to home
            uint32_t region_id = msg.page_addr.region_id;
            uint64_t page_idx  = page_index(msg.page_addr.offset);

            DirectoryEntry& entry = directory_.get_or_create(region_id, page_idx);
            entry.state = CoherenceState::UNCACHED;
            entry.epoch++;
            break;
        }

        case CoherenceOp::INVAL_ACK: {
            // Sharer acknowledged invalidation
            // Nothing to do — state was already updated in invalidate_sharers
            break;
        }

        case CoherenceOp::BARRIER_ENTER: {
            uint32_t count = barrier_count_.fetch_add(1) + 1;
            if (count >= static_cast<uint32_t>(num_nodes_ - 1)) {
                // All agents entered — release them
                CoherenceMessage leave{};
                leave.op    = CoherenceOp::BARRIER_LEAVE;
                leave.epoch = global_epoch_.fetch_add(1);
                for (int i = 0; i < num_nodes_; i++) {
                    if (i == node_id_) continue;
                    send_message(i, leave);
                }
                barrier_count_ = 0;
            }
            break;
        }

        case CoherenceOp::BARRIER_LEAVE: {
            // Agent received barrier release — advance epoch
            global_epoch_.fetch_add(1);
            break;
        }

        default: {
            DISTRO_WARN("Coherence: unhandled message op=%u from node %u",
                        static_cast<uint32_t>(msg.op), sender);
            break;
        }
    }
}

void CoherenceProtocol::invalidate_sharers(uint32_t region_id,
                                            uint64_t page_idx) {
    DirectoryEntry& entry = directory_.get_or_create(region_id, page_idx);

    CoherenceMessage inv{};
    inv.op    = CoherenceOp::INVALIDATE;
    inv.epoch = entry.epoch;

    for (int i = 0; i < num_nodes_; i++) {
        if (i == node_id_) continue;
        if (entry.has_sharer(i)) {
            send_message(i, inv);
        }
    }
}

// ── Message delivery ─────────────────────────────────────────────────────

void CoherenceProtocol::send_message(uint16_t target,
                                       const CoherenceMessage& msg) {
    auto it = msg_bufs_.find(target);
    DISTRO_CHECK(it != msg_bufs_.end(),
                 "Coherence: no message buffer for node " +
                 std::to_string(target));

    const MsgBuf& mb = it->second;
    GlobalAddress dst{target, mb.region_id, 0};

    dmem_->put_with_imm(target, dst, &msg, sizeof(CoherenceMessage),
                        static_cast<uint32_t>(msg.op));
}

void CoherenceProtocol::send_message_with_data(uint16_t target,
                                                CoherenceOp op,
                                                GlobalAddress page,
                                                uint64_t epoch) {
    CoherenceMessage msg{};
    msg.op        = op;
    msg.page_addr = page;
    msg.epoch     = epoch;
    msg.sender_seq = global_epoch_.load();
    send_message(target, msg);
}

// ── Local cache tracking ─────────────────────────────────────────────────

CoherenceProtocol::CachedPage*
CoherenceProtocol::find_cached(GlobalAddress page) {
    auto it = local_cache_.find(page.pack());
    if (it != local_cache_.end()) {
        return &it->second;
    }
    return nullptr;
}

void CoherenceProtocol::track_page(GlobalAddress page,
                                     CoherenceState state,
                                     uint64_t epoch) {
    CachedPage cp{};
    cp.gaddr       = page;
    cp.local_state = state;
    cp.epoch       = epoch;
    cp.dirty       = false;
    local_cache_[page.pack()] = cp;
}

void CoherenceProtocol::untrack_page(GlobalAddress page) {
    local_cache_.erase(page.pack());
}

} // namespace distro
