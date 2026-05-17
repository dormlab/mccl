#pragma once

#include "runtime/Scheduler.hpp"
#include "dmem/MemoryCatalog.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <chrono>

namespace mccl {

static constexpr int MAX_CLUSTER_NODES = 8;

enum class NodeStatus : uint8_t {
    UNKNOWN  = 0,
    ONLINE   = 1,
    DEGRADED = 2,
    OFFLINE  = 3,
};

/// Information about a single node in the cluster.
struct NodeInfo {
    uint16_t   node_id = 0;
    std::string hostname;
    std::string ip_address;
    uint16_t   control_port = 9800;
    uint64_t   total_memory = 0;
    uint64_t   free_memory  = 0;
    double     compute_capacity = 1.0;
    NodeStatus status = NodeStatus::UNKNOWN;
    uint64_t   last_heartbeat_us = 0;
    uint64_t   heartbeat_seq = 0;
};

/// Replicated cluster state — consistent across all online nodes.
struct ClusterState {
    std::array<NodeInfo, MAX_CLUSTER_NODES> nodes;
    uint64_t version = 0;  // Monotonically increasing on membership change
    uint16_t leader_id = 0;
    int      online_count = 0;
};

/// Cluster Manager — runs on the head node.
///
/// Responsibilities:
///   - TCP heartbeat protocol (active probes, not passive)
///   - Node membership (join, leave, dead detection)
///   - Leader election (lowest online node_id)
///   - Region catalog replication
///   - Job scheduler integration
///   - Fault tolerance (rebuild state on node failure)
///
/// Threads:
///   1. Heartbeat sender — sends PING to every agent every 500ms
///   2. Heartbeat monitor — checks for missed responses, marks nodes OFFLINE
///   3. Scheduler tick — calls schedule_tick() every 1s
class ClusterManager {
public:
    struct Config {
        uint16_t   node_id       = 0;
        int        total_nodes   = 3;
        uint16_t   control_port  = 9800;
        int        heartbeat_interval_ms = 500;
        int        heartbeat_timeout_ms  = 2000;  // 4 missed → OFFLINE
        int        scheduler_tick_ms     = 1000;
        uint64_t   memory_per_node = 0;
        bool       is_head        = true;
    };

    ClusterManager(const Config& config);
    ~ClusterManager();

    ClusterManager(const ClusterManager&) = delete;
    ClusterManager& operator=(const ClusterManager&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────
    void start();
    void shutdown();

    // ── Node management ──────────────────────────────────────────────
    void handle_join(NodeInfo info);
    void handle_leave(uint16_t node_id);
    void handle_heartbeat(const NodeInfo& info);

    // ── Leader election ──────────────────────────────────────────────
    bool is_leader() const;
    uint16_t leader_id() const;
    void elect_leader();

    // ── Health ───────────────────────────────────────────────────────
    std::vector<uint16_t> online_nodes() const;
    const NodeInfo* node_info(uint16_t node_id) const;
    ClusterState get_state() const;
    bool is_node_alive(uint16_t node_id) const;

    // ── Region catalog ───────────────────────────────────────────────
    void publish_region(uint16_t node_id, uint32_t region_id,
                        uint64_t local_addr, uint64_t length, uint32_t rkey);
    void unpublish_region(uint16_t node_id, uint32_t region_id);
    const RegionDirectoryEntry* lookup_region(uint16_t node_id,
                                               uint32_t region_id) const;

    // ── Job scheduling ───────────────────────────────────────────────
    uint64_t submit_job(const std::string& name, const std::string& owner,
                        uint16_t nodes, uint64_t memory_per_node,
                        JobPriority priority = JobPriority::NORMAL);
    bool     cancel_job(uint64_t job_id);
    bool     complete_job(uint64_t job_id, bool success = true);
    std::vector<JobDescriptor> pending_jobs() const;
    std::vector<JobDescriptor> active_jobs() const;

    // ── Fault tolerance ──────────────────────────────────────────────
    void handle_node_failure(uint16_t failed_node_id);

private:
    // ── Heartbeat loop ───────────────────────────────────────────────
    void heartbeat_sender_loop();
    void heartbeat_monitor_loop();
    void scheduler_loop();

    // ── Config ───────────────────────────────────────────────────────
    uint16_t node_id_;
    int      total_nodes_;
    uint16_t control_port_;
    int      heartbeat_interval_ms_;
    int      heartbeat_timeout_ms_;
    int      scheduler_tick_ms_;
    bool     is_head_;

    // ── Node state ───────────────────────────────────────────────────
    ClusterState state_;
    mutable std::mutex state_mu_;

    // ── Heartbeat tracking ───────────────────────────────────────────
    std::array<uint64_t, MAX_CLUSTER_NODES> last_heartbeat_us_{};

    // ── Region catalog (replicated) ──────────────────────────────────
    MemoryCatalog catalog_;
    mutable std::mutex catalog_mu_;

    // ── Scheduler ────────────────────────────────────────────────────
    std::unique_ptr<Scheduler> scheduler_;

    // ── Threads ──────────────────────────────────────────────────────
    std::thread heartbeat_sender_;
    std::thread heartbeat_monitor_;
    std::thread scheduler_thread_;

    std::atomic<bool> running_{false};
};

} // namespace mccl
