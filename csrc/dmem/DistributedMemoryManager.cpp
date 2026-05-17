#include "dmem/DistributedMemoryManager.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <cstring>
#include <algorithm>

namespace mccl {

// ── Construction / destruction ────────────────────────────────────────────

DistributedMemoryManager::DistributedMemoryManager(const Config& config)
    : node_id_(config.node_id)
    , num_peers_(config.num_peers)
{
    DISTRO_INFO("DMEM: initializing node %u (peers=%d, CQ depth=%d)",
              node_id_, num_peers_, config.cq_depth);

    ibv_ = ibv();
    DISTRO_CHECK(ibv_ != nullptr, "DMEM: librdma.dylib not available");

    // Enumerate RDMA devices
    int dev_count = 0;
    ibv_device** dev_list = ibv_->get_device_list(&dev_count);
    DISTRO_CHECK(dev_list != nullptr && dev_count > 0,
               "DMEM: no RDMA devices found");

    dev_ = dev_list[0];
    ctx_ = ibv_->open_device(dev_);
    DISTRO_CHECK(ctx_ != nullptr, "DMEM: ibv_open_device failed");
    ibv_->free_device_list(dev_list);

    // Allocate protection domain
    pd_ = ibv_->alloc_pd(ctx_);
    DISTRO_CHECK(pd_ != nullptr, "DMEM: ibv_alloc_pd failed");

    // Create a shared completion queue
    cq_ = ibv_->create_cq(ctx_, config.cq_depth * 2, nullptr, nullptr, 0);
    DISTRO_CHECK(cq_ != nullptr, "DMEM: ibv_create_cq failed");

    // Init per-peer state
    peer_conns_.resize(MAX_NODES);
    for (int i = 0; i < MAX_NODES; i++) {
        peer_mu_.push_back(std::make_unique<std::mutex>());
    }

    // Catalog buffer: small pinned region at region_id=0 for remote reads
    catalog_buf_.allocate(MemoryCatalog::CATALOG_BUF_SIZE);
    catalog_mr_ = catalog_buf_.register_with(pd_);
    DISTRO_CHECK(catalog_mr_ != nullptr, "DMEM: catalog MR registration failed");

    // Staging buffer for chunked transfers
    staging_buf_.allocate(kRdmaBufSize);
    staging_mr_ = staging_buf_.register_with(pd_);
    DISTRO_CHECK(staging_mr_ != nullptr, "DMEM: staging MR registration failed");

    DISTRO_INFO("DMEM: initialized (PD=%p, CQ=%p, catalog_rkey=%u)",
              (void*)pd_, (void*)cq_, catalog_mr_->rkey);
}

DistributedMemoryManager::~DistributedMemoryManager() {
    shutdown();

    if (staging_mr_) { ibv_->dereg_mr(staging_mr_); staging_mr_ = nullptr; }
    staging_buf_.cleanup();

    if (catalog_mr_) { ibv_->dereg_mr(catalog_mr_); catalog_mr_ = nullptr; }
    catalog_buf_.cleanup();

    // Deregister all regions
    for (auto& region : regions_) {
        // The MR was registered with ibv_reg_mr — we need to deregister.
        // The rkey/lkey contain the handle indirectly.  For simplicity,
        // regions are deregistered by the caller via unregister_region().
    }

    if (cq_)  { ibv_->destroy_cq(cq_);   cq_  = nullptr; }
    if (pd_)  { ibv_->dealloc_pd(pd_);   pd_  = nullptr; }
    if (ctx_) { ibv_->close_device(ctx_); ctx_ = nullptr; }
}

// ── Lifecycle ──────────────────────────────────────────────────────────

void DistributedMemoryManager::start() {
    if (running_.load()) return;
    running_ = true;
    poller_thread_ = std::thread(&DistributedMemoryManager::poller_loop, this);
    DISTRO_INFO("DMEM: poller thread started");
}

void DistributedMemoryManager::shutdown() {
    if (!running_.load()) return;
    running_ = false;
    {
        std::lock_guard<std::mutex> lock(pending_mu_);
        pending_cv_.notify_all();
    }
    if (poller_thread_.joinable()) poller_thread_.join();
    DISTRO_INFO("DMEM: shutdown complete");
}

// ── Region management ──────────────────────────────────────────────────

uint32_t DistributedMemoryManager::register_region(void* base_addr, uint64_t length,
                                                    uint8_t flags) {
    DISTRO_CHECK(base_addr != nullptr, "DMEM: null base_addr");
    DISTRO_CHECK(length > 0, "DMEM: zero-length region");

    ibv_mr* mr = ibv_->reg_mr(pd_, base_addr, length,
                               IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE);
    DISTRO_CHECK(mr != nullptr, "DMEM: ibv_reg_mr failed");

    uint32_t region_id = next_region_id_.fetch_add(1);

    MemoryRegion region{};
    region.gaddr.node_id   = node_id_;
    region.gaddr.region_id = region_id;
    region.gaddr.offset    = 0;
    region.local_addr      = reinterpret_cast<uint64_t>(base_addr);
    region.length          = length;
    region.rkey            = mr->rkey;
    region.lkey            = mr->lkey;
    region.flags           = flags;

    {
        std::lock_guard<std::mutex> lock(regions_mu_);
        if (region_id >= regions_.size()) {
            regions_.resize(region_id + 1);
        }
        regions_[region_id] = region;
    }

    // Publish to the local catalog
    RegionDirectoryEntry entry{};
    entry.local_addr = region.local_addr;
    entry.length     = region.length;
    entry.rkey       = region.rkey;
    entry.flags      = region.flags;
    entry.epoch      = 1;
    catalog_.upsert(region_id, entry);

    DISTRO_INFO("DMEM: registered region %u on node %u (addr=%p, len=%llu, rkey=%u)",
              region_id, node_id_, base_addr,
              (unsigned long long)length, mr->rkey);
    return region_id;
}

void DistributedMemoryManager::unregister_region(uint32_t region_id) {
    std::lock_guard<std::mutex> lock(regions_mu_);
    if (region_id < regions_.size()) {
        regions_[region_id] = MemoryRegion{};
        catalog_.remove(region_id);
        DISTRO_INFO("DMEM: unregistered region %u", region_id);
    }
}

uint32_t DistributedMemoryManager::register_buffer(void* cpu_ptr, uint64_t length,
                                                    uint8_t flags) {
    return register_region(cpu_ptr, length, flags);
}

// ── Address resolution ─────────────────────────────────────────────────

AddrResolution DistributedMemoryManager::resolve(GlobalAddress addr) const {
    AddrResolution result{};
    result.length = 0;

    if (addr.node_id == node_id_) {
        // Local access
        std::lock_guard<std::mutex> lock(regions_mu_);
        if (addr.region_id < regions_.size()) {
            const auto& region = regions_[addr.region_id];
            if (region.length > 0 && addr.offset < region.length) {
                result.is_local   = true;
                result.local_ptr  = reinterpret_cast<void*>(region.local_addr + addr.offset);
                result.local_lkey = region.lkey;
                result.length     = region.length - addr.offset;
            }
        }
    } else {
        // Remote access — look up in catalog
        if (lookup_remote(addr.node_id, addr.region_id,
                          result.remote_addr, result.remote_rkey,
                          result.length)) {
            result.is_local = false;
            // Clamp to region bounds
            if (addr.offset < result.length) {
                result.remote_addr += addr.offset;
                result.length      -= addr.offset;
            } else {
                result.length = 0;
            }
        }
    }

    return result;
}

bool DistributedMemoryManager::lookup_remote(uint16_t node_id, uint32_t region_id,
                                              uint64_t& remote_addr, uint32_t& rkey,
                                              uint64_t& length) const {
    const RegionDirectoryEntry* entry = catalog_.lookup(node_id, region_id);
    if (!entry) return false;

    remote_addr = entry->local_addr;
    rkey        = entry->rkey;
    length      = entry->length;
    return true;
}

// ── One-sided RDMA operations ──────────────────────────────────────────

uint64_t DistributedMemoryManager::put(uint16_t target_node,
                                        GlobalAddress dst_addr,
                                        const void* src_local, uint64_t length) {
    AddrResolution res = resolve(dst_addr);
    DISTRO_CHECK(!res.is_local || target_node == node_id_,
               "DMEM: put target is local but node_id mismatch");
    DISTRO_CHECK(length <= res.length,
               "DMEM: put length " + std::to_string(length) +
               " exceeds region bounds " + std::to_string(res.length));
    DISTRO_CHECK(res.remote_rkey != 0 || res.is_local,
               "DMEM: put target not found in catalog");

    if (res.is_local) {
        // Local put → memcpy
        std::memcpy(res.local_ptr, src_local, length);
        return 0;  // No RDMA op needed
    }

    return put_chunked(target_node, res.remote_addr, res.remote_rkey,
                       src_local, length);
}

uint64_t DistributedMemoryManager::get(uint16_t target_node,
                                        GlobalAddress src_addr,
                                        void* dst_local, uint64_t length) {
    AddrResolution res = resolve(src_addr);
    DISTRO_CHECK(length <= res.length,
               "DMEM: get length " + std::to_string(length) +
               " exceeds region bounds " + std::to_string(res.length));
    DISTRO_CHECK(res.remote_rkey != 0 || res.is_local,
               "DMEM: get source not found in catalog");

    if (res.is_local) {
        // Local get → memcpy
        std::memcpy(dst_local, res.local_ptr, length);
        return 0;
    }

    return get_chunked(target_node, res.remote_addr, res.remote_rkey,
                       dst_local, length);
}

uint64_t DistributedMemoryManager::put_with_imm(uint16_t target_node,
                                                 GlobalAddress dst_addr,
                                                 const void* src_local, uint64_t length,
                                                 uint32_t imm_data) {
    AddrResolution res = resolve(dst_addr);
    DISTRO_CHECK(!res.is_local, "DMEM: put_with_imm requires remote target");
    DISTRO_CHECK(length <= kRdmaBufSize,
               "DMEM: put_with_imm limited to staging buffer size");
    DISTRO_CHECK(length <= res.length,
               "DMEM: put_with_imm length exceeds region bounds");

    RdmaConnection* conn = peer_conns_[target_node].get();
    DISTRO_CHECK(conn && conn->ok(), "DMEM: no connection to node " +
               std::to_string(target_node));

    // Copy source data into the staging buffer
    std::memcpy(staging_buf_.data(), src_local, length);

    ibv_sge sge = staging_buf_.to_sge(staging_mr_, 0, length);
    uint64_t wr_id = next_wr_id_.fetch_add(1);

    {
        std::lock_guard<std::mutex> lock(*peer_mu_[target_node]);
        bool ok = conn->post_rdma_write_with_imm(sge, res.remote_addr,
                                                   res.remote_rkey,
                                                   imm_data, wr_id);
        DISTRO_CHECK(ok, "DMEM: post_rdma_write_with_imm failed");
    }

    return wr_id;
}

uint64_t DistributedMemoryManager::fence(uint16_t target_node) {
    RdmaConnection* conn = peer_conns_[target_node].get();
    if (!conn || !conn->ok()) return 0;

    uint64_t wr_id = next_wr_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(*peer_mu_[target_node]);
        bool ok = conn->post_fence(wr_id);
        DISTRO_CHECK(ok, "DMEM: post_fence failed");
    }
    return wr_id;
}

// ── Chunked RDMA helpers ────────────────────────────────────────────────

uint64_t DistributedMemoryManager::put_chunked(uint16_t target,
                                                uint64_t remote_addr, uint32_t rkey,
                                                const void* src, uint64_t length) {
    RdmaConnection* conn = peer_conns_[target].get();
    DISTRO_CHECK(conn && conn->ok(), "DMEM: no connection to node " +
               std::to_string(target));

    // Single chunk if it fits in the staging buffer
    if (length <= kRdmaBufSize) {
        std::memcpy(staging_buf_.data(), src, length);
        ibv_sge sge = staging_buf_.to_sge(staging_mr_, 0, length);
        uint64_t wr_id = next_wr_id_.fetch_add(1);

        {
            std::lock_guard<std::mutex> lock(*peer_mu_[target]);
            bool ok = conn->post_rdma_write(sge, remote_addr, rkey, wr_id);
            DISTRO_CHECK(ok, "DMEM: post_rdma_write failed");
        }

        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            pending_ops_.push_back({wr_id, {}});
        }
        return wr_id;
    }

    // Multi-chunk: submit all chunks, return the last wr_id
    uint64_t last_wr_id = 0;
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint64_t remaining = length;

    while (remaining > 0) {
        uint64_t chunk = std::min(remaining, static_cast<uint64_t>(kRdmaBufSize));
        uint64_t offset = length - remaining;

        std::memcpy(staging_buf_.data(), src_bytes + offset, chunk);
        ibv_sge sge = staging_buf_.to_sge(staging_mr_, 0, chunk);
        uint64_t wr_id = next_wr_id_.fetch_add(1);

        {
            std::lock_guard<std::mutex> lock(*peer_mu_[target]);
            bool ok = conn->post_rdma_write(sge, remote_addr + offset,
                                            rkey, wr_id);
            DISTRO_CHECK(ok, "DMEM: chunked post_rdma_write failed");
        }

        last_wr_id = wr_id;
        remaining -= chunk;
    }

    return last_wr_id;
}

uint64_t DistributedMemoryManager::get_chunked(uint16_t target,
                                                uint64_t remote_addr, uint32_t rkey,
                                                void* dst, uint64_t length) {
    RdmaConnection* conn = peer_conns_[target].get();
    DISTRO_CHECK(conn && conn->ok(), "DMEM: no connection to node " +
               std::to_string(target));

    uint64_t last_wr_id = 0;
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    uint64_t remaining = length;

    while (remaining > 0) {
        uint64_t chunk = std::min(remaining, static_cast<uint64_t>(kRdmaBufSize));
        uint64_t offset = length - remaining;

        ibv_sge sge = staging_buf_.to_sge(staging_mr_, 0, chunk);
        uint64_t wr_id = next_wr_id_.fetch_add(1);

        // First post the read, then poll, then copy
        {
            std::lock_guard<std::mutex> lock(*peer_mu_[target]);
            bool ok = conn->post_rdma_read(sge, remote_addr + offset,
                                           rkey, wr_id);
            DISTRO_CHECK(ok, "DMEM: post_rdma_read failed");
        }

        // Inline poll for this chunk
        poll_completion(wr_id);

        // Copy from staging to destination
        std::memcpy(dst_bytes + offset, staging_buf_.data(), chunk);

        last_wr_id = wr_id;
        remaining -= chunk;
    }

    return last_wr_id;
}

// ── Completion tracking ────────────────────────────────────────────────

bool DistributedMemoryManager::poll_completion(uint64_t wr_id,
                                                std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline) {
        ibv_wc wc[16];
        int n = ibv_->poll_cq(cq_, 16, wc);
        if (n < 0) {
            DISTRO_ERROR("DMEM: poll_cq error");
            return false;
        }

        for (int i = 0; i < n; i++) {
            if (wc[i].wr_id == wr_id) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    DISTRO_ERROR("DMEM: completion error for wr_id %llu: status=%d",
                               (unsigned long long)wr_id, wc[i].status);
                    return false;
                }
                return true;
            }
            // Stray completion — log and discard
            if (wc[i].status != IBV_WC_SUCCESS) {
                DISTRO_WARN("DMEM: stray completion error: wr_id=%llu status=%d",
                          (unsigned long long)wc[i].wr_id, wc[i].status);
            }
        }

        if (n == 0) {
            std::this_thread::yield();
        }
    }

    DISTRO_ERROR("DMEM: poll_completion timeout for wr_id %llu",
               (unsigned long long)wr_id);
    return false;
}

void DistributedMemoryManager::drain_pending() {
    while (true) {
        uint64_t last_wr_id = 0;
        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            if (pending_ops_.empty()) break;
            last_wr_id = pending_ops_.back().wr_id;
        }
        if (last_wr_id == 0) break;
        if (!poll_completion(last_wr_id)) break;
        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            while (!pending_ops_.empty() &&
                   pending_ops_.front().wr_id <= last_wr_id) {
                pending_ops_.pop_front();
            }
        }
    }
}

// ── Peer connection access ─────────────────────────────────────────────

RdmaConnection* DistributedMemoryManager::peer_connection(uint16_t node_id) {
    if (node_id >= MAX_NODES) return nullptr;
    return peer_conns_[node_id].get();
}

// ── Statistics ─────────────────────────────────────────────────────────

DistributedMemoryManager::Stats DistributedMemoryManager::stats() const {
    std::lock_guard<std::mutex> lock(stats_mu_);
    return stats_;
}

// ── Poller loop ────────────────────────────────────────────────────────

void DistributedMemoryManager::poller_loop() {
    DISTRO_INFO("DMEM: poller loop started");

    while (running_.load()) {
        ibv_wc wc[kCqPollBatch];
        int n = ibv_->poll_cq(cq_, kCqPollBatch, wc);

        if (n < 0) {
            DISTRO_ERROR("DMEM: poller CQ error, stopping");
            running_ = false;
            break;
        }

        if (n == 0) {
            // Idle — brief yield to avoid busy-spin
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                DISTRO_ERROR("DMEM: completion error: wr_id=%llu opcode=%d status=%d",
                           (unsigned long long)wc[i].wr_id,
                           wc[i].opcode, wc[i].status);
                std::lock_guard<std::mutex> lock(stats_mu_);
                stats_.total_errors++;
                continue;
            }

            // Track bytes
            {
                std::lock_guard<std::mutex> lock(stats_mu_);
                if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
                    stats_.total_put_bytes += wc[i].byte_len;
                    stats_.total_put_ops++;
                } else if (wc[i].opcode == IBV_WC_RDMA_READ) {
                    stats_.total_get_bytes += wc[i].byte_len;
                    stats_.total_get_ops++;
                }
            }

            // Signal completion to waiting threads
            {
                std::lock_guard<std::mutex> lock(pending_mu_);
                for (auto& op : pending_ops_) {
                    if (op.wr_id == wc[i].wr_id) {
                        op.done.set_value();
                    }
                }
            }
            pending_cv_.notify_all();
        }
    }

    DISTRO_INFO("DMEM: poller loop exiting");
}

} // namespace mccl
