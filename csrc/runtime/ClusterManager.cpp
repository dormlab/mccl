#include "runtime/ClusterManager.hpp"
#include "agent/Protocol.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace distro {

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

// ── Construction / destruction ──────────────────────────────────────────────

ClusterManager::ClusterManager(const Config& config)
    : node_id_(config.node_id)
    , total_nodes_(config.total_nodes)
    , control_port_(config.control_port)
    , heartbeat_interval_ms_(config.heartbeat_interval_ms)
    , heartbeat_timeout_ms_(config.heartbeat_timeout_ms)
    , scheduler_tick_ms_(config.scheduler_tick_ms)
    , is_head_(config.is_head)
{
    // Init local node info
    state_.nodes[node_id_].node_id   = node_id_;
    state_.nodes[node_id_].status    = NodeStatus::ONLINE;
    state_.nodes[node_id_].hostname  = "localhost";
    state_.nodes[node_id_].control_port = control_port_;
    state_.leader_id = node_id_;  // Assume leader until election
    state_.online_count = 1;

    // Init scheduler
    Scheduler::Config sched_cfg{};
    sched_cfg.total_nodes     = total_nodes_;
    sched_cfg.memory_per_node = config.memory_per_node;
    sched_cfg.tick_interval_ms = scheduler_tick_ms_;
    scheduler_ = std::make_unique<Scheduler>(sched_cfg);

    DISTRO_INFO("ClusterManager: node %u initialized (head=%s, total=%d)",
                node_id_, is_head_ ? "yes" : "no", total_nodes_);
}

ClusterManager::~ClusterManager() { shutdown(); }

void ClusterManager::start() {
    if (running_.load()) return;
    running_ = true;

    if (is_head_) {
        heartbeat_sender_  = std::thread(&ClusterManager::heartbeat_sender_loop, this);
        heartbeat_monitor_ = std::thread(&ClusterManager::heartbeat_monitor_loop, this);
        scheduler_thread_  = std::thread(&ClusterManager::scheduler_loop, this);
    }

    scheduler_->start();
    DISTRO_INFO("ClusterManager: started");
}

void ClusterManager::shutdown() {
    if (!running_.load()) return;
    running_ = false;

    if (heartbeat_sender_.joinable())  heartbeat_sender_.join();
    if (heartbeat_monitor_.joinable()) heartbeat_monitor_.join();
    if (scheduler_thread_.joinable())  scheduler_thread_.join();
    scheduler_->shutdown();

    DISTRO_INFO("ClusterManager: shutdown complete");
}

// ── Node management ────────────────────────────────────────────────────────

void ClusterManager::handle_join(NodeInfo info) {
    std::lock_guard<std::mutex> lock(state_mu_);

    auto& entry = state_.nodes[info.node_id];
    entry = info;
    entry.status = NodeStatus::ONLINE;
    entry.last_heartbeat_us = now_us();
    state_.online_count++;
    state_.version++;

    elect_leader();

    DISTRO_INFO("ClusterManager: node %u joined (%s, total=%d)",
                info.node_id, info.hostname.c_str(), state_.online_count);
}

void ClusterManager::handle_leave(uint16_t node_id) {
    std::lock_guard<std::mutex> lock(state_mu_);

    if (state_.nodes[node_id].status == NodeStatus::OFFLINE) return;

    state_.nodes[node_id].status = NodeStatus::OFFLINE;
    state_.online_count--;
    state_.version++;

    // Clear catalog entries for this node
    {
        std::lock_guard<std::mutex> cl(catalog_mu_);
        catalog_.remove_node(node_id);
    }

    elect_leader();

    // If the leader left, handle it
    if (node_id == state_.leader_id) {
        handle_node_failure(node_id);
    }

    DISTRO_WARN("ClusterManager: node %u left (online=%d)",
                node_id, state_.online_count);
}

void ClusterManager::handle_heartbeat(const NodeInfo& info) {
    std::lock_guard<std::mutex> lock(state_mu_);

    auto& entry = state_.nodes[info.node_id];
    if (entry.status == NodeStatus::UNKNOWN) {
        entry = info;
        entry.status = NodeStatus::ONLINE;
    }

    entry.last_heartbeat_us = now_us();
    entry.heartbeat_seq     = info.heartbeat_seq;
    entry.free_memory       = info.free_memory;

    if (entry.status != NodeStatus::ONLINE) {
        entry.status = NodeStatus::ONLINE;
        state_.online_count++;
        state_.version++;
    }
}

// ── Leader election ────────────────────────────────────────────────────────

bool ClusterManager::is_leader() const {
    std::lock_guard<std::mutex> lock(state_mu_);
    return state_.leader_id == node_id_;
}

uint16_t ClusterManager::leader_id() const {
    std::lock_guard<std::mutex> lock(state_mu_);
    return state_.leader_id;
}

void ClusterManager::elect_leader() {
    // Lowest online node_id wins
    for (int i = 0; i < MAX_CLUSTER_NODES; i++) {
        if (state_.nodes[i].status == NodeStatus::ONLINE) {
            state_.leader_id = static_cast<uint16_t>(i);
            return;
        }
    }
    state_.leader_id = node_id_;  // Fallback to self
}

// ── Health ─────────────────────────────────────────────────────────────────

std::vector<uint16_t> ClusterManager::online_nodes() const {
    std::lock_guard<std::mutex> lock(state_mu_);
    std::vector<uint16_t> result;
    for (int i = 0; i < MAX_CLUSTER_NODES; i++) {
        if (state_.nodes[i].status == NodeStatus::ONLINE) {
            result.push_back(static_cast<uint16_t>(i));
        }
    }
    return result;
}

const NodeInfo* ClusterManager::node_info(uint16_t node_id) const {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (node_id < MAX_CLUSTER_NODES &&
        state_.nodes[node_id].status != NodeStatus::UNKNOWN) {
        return &state_.nodes[node_id];
    }
    return nullptr;
}

ClusterState ClusterManager::get_state() const {
    std::lock_guard<std::mutex> lock(state_mu_);
    return state_;
}

bool ClusterManager::is_node_alive(uint16_t node_id) const {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (node_id >= MAX_CLUSTER_NODES) return false;
    return state_.nodes[node_id].status == NodeStatus::ONLINE;
}

// ── Region catalog ─────────────────────────────────────────────────────────

void ClusterManager::publish_region(uint16_t node_id, uint32_t region_id,
                                     uint64_t local_addr, uint64_t length,
                                     uint32_t rkey) {
    std::lock_guard<std::mutex> lock(catalog_mu_);

    RegionDirectoryEntry entry{};
    entry.local_addr = local_addr;
    entry.length     = length;
    entry.rkey       = rkey;
    entry.flags      = MemoryRegion::READABLE | MemoryRegion::WRITABLE;
    entry.epoch      = 1;

    catalog_.upsert(region_id, entry);

    DISTRO_DEBUG("ClusterManager: published region %u on node %u "
                 "(addr=%llx, len=%llu, rkey=%u)",
                 region_id, node_id,
                 (unsigned long long)local_addr,
                 (unsigned long long)length, rkey);
}

void ClusterManager::unpublish_region(uint16_t node_id, uint32_t region_id) {
    std::lock_guard<std::mutex> lock(catalog_mu_);
    catalog_.remove(region_id);
    (void)node_id;
}

const RegionDirectoryEntry*
ClusterManager::lookup_region(uint16_t node_id, uint32_t region_id) const {
    std::lock_guard<std::mutex> lock(catalog_mu_);
    return catalog_.lookup(node_id, region_id);
}

// ── Job scheduling ─────────────────────────────────────────────────────────

uint64_t ClusterManager::submit_job(const std::string& name,
                                     const std::string& owner,
                                     uint16_t nodes, uint64_t memory_per_node,
                                     JobPriority priority) {
    JobDescriptor job{};
    job.name             = name;
    job.owner            = owner;
    job.requested_nodes  = nodes;
    job.memory_per_node  = memory_per_node;
    job.priority         = priority;
    return scheduler_->submit(job);
}

bool ClusterManager::cancel_job(uint64_t job_id) {
    return scheduler_->cancel(job_id);
}

bool ClusterManager::complete_job(uint64_t job_id, bool success) {
    return scheduler_->complete(job_id, success);
}

std::vector<JobDescriptor> ClusterManager::pending_jobs() const {
    return scheduler_->pending_jobs();
}

std::vector<JobDescriptor> ClusterManager::active_jobs() const {
    return scheduler_->active_jobs();
}

// ── Fault tolerance ────────────────────────────────────────────────────────

void ClusterManager::handle_node_failure(uint16_t failed_node_id) {
    DISTRO_ERROR("ClusterManager: node %u FAILED", failed_node_id);

    // Mark offline
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        state_.nodes[failed_node_id].status = NodeStatus::OFFLINE;
        state_.online_count--;
        state_.version++;
    }

    // Clear catalog entries
    {
        std::lock_guard<std::mutex> lock(catalog_mu_);
        catalog_.remove_node(failed_node_id);
    }

    // Evict affected jobs
    scheduler_->evict_node(failed_node_id);

    // Re-elect if the leader failed
    if (is_leader()) {
        elect_leader();
    }

    DISTRO_WARN("ClusterManager: node %u failure handled (online=%d)",
                failed_node_id, state_.online_count);
}

// ── Heartbeat loops ────────────────────────────────────────────────────────

void ClusterManager::heartbeat_sender_loop() {
    DISTRO_INFO("ClusterManager: heartbeat sender started (interval=%dms)",
                heartbeat_interval_ms_);

    while (running_.load()) {
        auto online = online_nodes();

        for (uint16_t nid : online) {
            if (nid == node_id_) continue;  // Don't ping self

            const NodeInfo* info = node_info(nid);
            if (!info) continue;

            // Send PING over TCP to the agent
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) continue;

            struct sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port   = htons(info->control_port);
            inet_pton(AF_INET, info->ip_address.c_str(), &addr.sin_addr);

            struct timeval tv{1, 0};  // 1 second connect timeout
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

            if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                AgentHeader hdr{AgentCmd::PING, AgentStatus::OK, 0};
                ::send(sock, &hdr, sizeof(hdr), 0);

                AgentHeader resp{};
                if (::recv(sock, &resp, sizeof(resp), 0) == sizeof(resp)) {
                    NodeInfo updated = *info;
                    updated.last_heartbeat_us = now_us();
                    updated.heartbeat_seq++;
                    handle_heartbeat(updated);
                }
            }
            close(sock);
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(heartbeat_interval_ms_));
    }
}

void ClusterManager::heartbeat_monitor_loop() {
    DISTRO_INFO("ClusterManager: heartbeat monitor started (timeout=%dms)",
                heartbeat_timeout_ms_);

    while (running_.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(heartbeat_interval_ms_));

        uint64_t now = now_us();
        std::lock_guard<std::mutex> lock(state_mu_);

        for (int i = 0; i < total_nodes_; i++) {
            if (i == node_id_) continue;
            auto& node = state_.nodes[i];
            if (node.status != NodeStatus::ONLINE &&
                node.status != NodeStatus::DEGRADED) continue;

            uint64_t elapsed_us = now - node.last_heartbeat_us;
            uint64_t timeout_us = static_cast<uint64_t>(heartbeat_timeout_ms_) * 1000;

            if (elapsed_us > timeout_us) {
                DISTRO_ERROR("ClusterManager: node %u heartbeat timeout "
                             "(%.1fs since last)",
                             node.node_id, elapsed_us / 1'000'000.0);
                // handle_leave inside handle_node_failure
                // (call outside lock to avoid deadlock)
                uint16_t failed = node.node_id;
                state_mu_.unlock();
                handle_node_failure(failed);
                state_mu_.lock();
            }
        }
    }
}

void ClusterManager::scheduler_loop() {
    DISTRO_INFO("ClusterManager: scheduler loop started (tick=%dms)",
                scheduler_tick_ms_);

    while (running_.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(scheduler_tick_ms_));

        auto online = online_nodes();

        // Build node load from active jobs
        std::vector<NodeLoad> loads(MAX_CLUSTER_NODES);
        for (auto nid : online) {
            const NodeInfo* info = node_info(nid);
            if (info) {
                loads[nid].memory_used = info->total_memory - info->free_memory;
            }
        }

        scheduler_->schedule_tick(online, loads);
    }
}

} // namespace distro
