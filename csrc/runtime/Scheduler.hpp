#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <chrono>

namespace mccl {

enum class JobPriority : uint8_t {
    LOW      = 0,
    NORMAL   = 1,
    HIGH     = 2,
    CRITICAL = 3,
};

enum class JobStatus : uint8_t {
    PENDING      = 0,
    RUNNING      = 1,
    COMPLETED    = 2,
    FAILED       = 3,
    RESCHEDULING = 4,
};

/// A submitted job — training run, inference task, or compute workload.
struct JobDescriptor {
    uint64_t    job_id = 0;
    std::string name;
    std::string owner;             // User or tenant identifier
    uint16_t    requested_nodes;   // How many nodes this job needs
    uint64_t    memory_per_node;   // Bytes of GPU memory required per node
    double      estimated_duration_s = 0;
    JobPriority priority = JobPriority::NORMAL;
    JobStatus   status   = JobStatus::PENDING;

    // Assigned resources (populated by scheduler)
    std::vector<uint16_t> assigned_nodes;

    // Timestamps (microseconds since epoch)
    uint64_t submitted_at_us = 0;
    uint64_t started_at_us   = 0;
    uint64_t completed_at_us = 0;
};

/// Per-node resource usage tracked by the scheduler.
struct NodeLoad {
    uint64_t memory_used = 0;
    double   compute_utilization = 0.0;
    int      active_jobs = 0;
};

/// Multi-tenant GPU job scheduler with priority queues and fair-share.
///
/// Runs a periodic tick (default 1 Hz) on the head node:
///   1. Sort queue by priority, then submission time (FCFS within priority)
///   2. Check resource availability (nodes, memory) against online nodes
///   3. Allocate matching jobs, transition PENDING → RUNNING
///   4. Track per-tenant node-time for fair-share boosting
///   5. HIGH/CRITICAL jobs can preempt LOW priority jobs
class Scheduler {
public:
    struct Config {
        int      tick_interval_ms = 1000;
        uint16_t total_nodes      = 3;
        uint64_t memory_per_node  = 0;  // 0 = auto-detect
        bool     preemption_enabled = true;
    };

    Scheduler(const Config& config);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────
    void start();
    void shutdown();

    // ── Job management ───────────────────────────────────────────────
    uint64_t submit(JobDescriptor job);
    bool     cancel(uint64_t job_id);
    bool     complete(uint64_t job_id, bool success = true);

    // ── Scheduling tick ──────────────────────────────────────────────
    /// Called periodically. Examines the queue and allocates resources.
    /// Returns the number of jobs started this tick.
    int schedule_tick(const std::vector<uint16_t>& online_nodes,
                      const std::vector<NodeLoad>& node_loads);

    // ── Queries ──────────────────────────────────────────────────────
    std::vector<JobDescriptor> pending_jobs() const;
    std::vector<JobDescriptor> active_jobs() const;
    std::vector<JobDescriptor> recent_history(int limit = 20) const;
    size_t queue_depth() const;
    size_t active_count() const;

    // ── Fair-share ───────────────────────────────────────────────────
    double tenant_share(const std::string& owner) const;

    // ── Fault recovery ───────────────────────────────────────────────
    /// Called when a node fails. Evicts affected jobs and re-queues them.
    void evict_node(uint16_t failed_node_id);

private:
    bool can_allocate(const JobDescriptor& job,
                      const std::vector<uint16_t>& online_nodes,
                      const std::vector<NodeLoad>& node_loads) const;

    void allocate(JobDescriptor& job,
                  const std::vector<uint16_t>& online_nodes,
                  std::vector<NodeLoad>& node_loads);

    Config config_;
    std::atomic<uint64_t> next_job_id_{1};

    // Priority-sorted queue (highest first)
    std::deque<JobDescriptor> queue_;
    std::vector<JobDescriptor> active_;
    std::vector<JobDescriptor> history_;

    // Per-tenant tracking for fair-share
    struct TenantStats {
        uint64_t total_node_seconds = 0;
        uint64_t last_scheduled_us  = 0;
    };
    std::unordered_map<std::string, TenantStats> tenant_stats_;

    mutable std::mutex mu_;
    std::atomic<bool> running_{false};
};

} // namespace mccl
