#include "runtime/Scheduler.hpp"
#include "common/Logging.hpp"

#include <algorithm>
#include <chrono>

namespace distro {

static uint64_t now_us() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

Scheduler::Scheduler(const Config& config) : config_(config) {
    DISTRO_INFO("Scheduler: initialized (nodes=%u, memory_per_node=%llu, "
                "preemption=%s)",
                config_.total_nodes,
                (unsigned long long)config_.memory_per_node,
                config_.preemption_enabled ? "on" : "off");
}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::start() { running_ = true; }
void Scheduler::shutdown() { running_ = false; }

// ── Job management ─────────────────────────────────────────────────────────

uint64_t Scheduler::submit(JobDescriptor job) {
    std::lock_guard<std::mutex> lock(mu_);

    job.job_id = next_job_id_.fetch_add(1);
    job.status = JobStatus::PENDING;
    job.submitted_at_us = now_us();

    queue_.push_back(job);

    DISTRO_INFO("Scheduler: job %llu submitted (\"%s\", priority=%d, nodes=%u)",
                (unsigned long long)job.job_id, job.name.c_str(),
                static_cast<int>(job.priority), job.requested_nodes);
    return job.job_id;
}

bool Scheduler::cancel(uint64_t job_id) {
    std::lock_guard<std::mutex> lock(mu_);

    // Remove from queue
    auto qit = std::find_if(queue_.begin(), queue_.end(),
        [job_id](const JobDescriptor& j) { return j.job_id == job_id; });
    if (qit != queue_.end()) {
        queue_.erase(qit);
        DISTRO_INFO("Scheduler: job %llu cancelled (was pending)", (unsigned long long)job_id);
        return true;
    }

    // Check active
    auto ait = std::find_if(active_.begin(), active_.end(),
        [job_id](const JobDescriptor& j) { return j.job_id == job_id; });
    if (ait != active_.end()) {
        ait->status = JobStatus::FAILED;
        ait->completed_at_us = now_us();
        history_.push_back(*ait);
        active_.erase(ait);
        DISTRO_INFO("Scheduler: job %llu cancelled (was running)", (unsigned long long)job_id);
        return true;
    }

    return false;
}

bool Scheduler::complete(uint64_t job_id, bool success) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = std::find_if(active_.begin(), active_.end(),
        [job_id](const JobDescriptor& j) { return j.job_id == job_id; });
    if (it == active_.end()) return false;

    it->status = success ? JobStatus::COMPLETED : JobStatus::FAILED;
    it->completed_at_us = now_us();

    // Update tenant stats
    if (it->started_at_us > 0) {
        uint64_t elapsed_s = (it->completed_at_us - it->started_at_us) / 1'000'000;
        tenant_stats_[it->owner].total_node_seconds +=
            elapsed_s * it->assigned_nodes.size();
    }

    history_.push_back(*it);
    active_.erase(it);

    DISTRO_INFO("Scheduler: job %llu %s",
                (unsigned long long)job_id,
                success ? "completed" : "failed");
    return true;
}

// ── Scheduling tick ────────────────────────────────────────────────────────

int Scheduler::schedule_tick(const std::vector<uint16_t>& online_nodes,
                              const std::vector<NodeLoad>& node_loads) {
    std::lock_guard<std::mutex> lock(mu_);
    int started = 0;

    // Sort queue: priority descending, then submission time ascending
    std::sort(queue_.begin(), queue_.end(),
              [](const JobDescriptor& a, const JobDescriptor& b) {
                  if (a.priority != b.priority)
                      return static_cast<int>(a.priority) > static_cast<int>(b.priority);
                  return a.submitted_at_us < b.submitted_at_us;
              });

    // Fair-share boost: tenants that have run less get a priority bump
    uint64_t total_node_s = 0;
    for (const auto& [owner, stats] : tenant_stats_) {
        total_node_s += stats.total_node_seconds;
    }

    if (total_node_s > 0) {
        for (auto& job : queue_) {
            auto it = tenant_stats_.find(job.owner);
            if (it != tenant_stats_.end()) {
                double share = static_cast<double>(it->second.total_node_seconds)
                               / std::max(total_node_s, 1ULL);
                // Boost jobs from tenants below fair share
                if (share < (1.0 / std::max(static_cast<size_t>(tenant_stats_.size()), 1UL))) {
                    // Effectively bump priority by one tier for fair-share
                    if (job.priority == JobPriority::LOW) {
                        job.priority = JobPriority::NORMAL;
                    }
                }
            }
        }
    }

    // Attempt to schedule each job
    auto it = queue_.begin();
    while (it != queue_.end()) {
        if (can_allocate(*it, online_nodes, node_loads)) {
            allocate(*it, online_nodes, const_cast<std::vector<NodeLoad>&>(node_loads));
            it->status = JobStatus::RUNNING;
            it->started_at_us = now_us();
            active_.push_back(*it);
            it = queue_.erase(it);
            started++;
        } else {
            ++it;
        }
    }

    // Preemption: HIGH/CRITICAL can evict LOW jobs if resources are tight
    if (config_.preemption_enabled) {
        for (auto& job : queue_) {
            if (job.priority < JobPriority::HIGH) continue;
            if (can_allocate(job, online_nodes, node_loads)) continue;

            // Try to free resources by evicting a LOW priority active job
            for (auto& active : active_) {
                if (active.priority == JobPriority::LOW &&
                    active.assigned_nodes.size() >= job.requested_nodes) {
                    DISTRO_INFO("Scheduler: preempting job %llu for %llu",
                                (unsigned long long)active.job_id,
                                (unsigned long long)job.job_id);

                    active.status = JobStatus::RESCHEDULING;
                    queue_.push_front(active);

                    // Free its nodes
                    auto nodes = active.assigned_nodes;
                    active.assigned_nodes.clear();
                    active = JobDescriptor{}; // mark for removal

                    // Try allocation again
                    if (can_allocate(job, online_nodes, node_loads)) {
                        allocate(job, online_nodes,
                                 const_cast<std::vector<NodeLoad>&>(node_loads));
                        job.status = JobStatus::RUNNING;
                        job.started_at_us = now_us();
                        active_.push_back(job);
                        started++;
                    }
                    break;
                }
            }

            // Clean up preempted jobs
            active_.erase(
                std::remove_if(active_.begin(), active_.end(),
                               [](const JobDescriptor& j) {
                                   return j.assigned_nodes.empty();
                               }),
                active_.end());
        }
    }

    if (started > 0) {
        DISTRO_DEBUG("Scheduler: started %d jobs this tick (queue=%zu, active=%zu)",
                     started, queue_.size(), active_.size());
    }

    return started;
}

// ── Queries ────────────────────────────────────────────────────────────────

std::vector<JobDescriptor> Scheduler::pending_jobs() const {
    std::lock_guard<std::mutex> lock(mu_);
    return std::vector<JobDescriptor>(queue_.begin(), queue_.end());
}

std::vector<JobDescriptor> Scheduler::active_jobs() const {
    std::lock_guard<std::mutex> lock(mu_);
    return active_;
}

std::vector<JobDescriptor> Scheduler::recent_history(int limit) const {
    std::lock_guard<std::mutex> lock(mu_);
    int count = std::min(limit, static_cast<int>(history_.size()));
    return std::vector<JobDescriptor>(
        history_.end() - count, history_.end());
}

size_t Scheduler::queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

size_t Scheduler::active_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return active_.size();
}

double Scheduler::tenant_share(const std::string& owner) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = tenant_stats_.find(owner);
    if (it == tenant_stats_.end()) return 0.0;

    uint64_t total = 0;
    for (const auto& [_, stats] : tenant_stats_) {
        total += stats.total_node_seconds;
    }
    if (total == 0) return 0.0;
    return static_cast<double>(it->second.total_node_seconds) / total;
}

// ── Fault recovery ─────────────────────────────────────────────────────────

void Scheduler::evict_node(uint16_t failed_node_id) {
    std::lock_guard<std::mutex> lock(mu_);

    for (auto& job : active_) {
        auto& nodes = job.assigned_nodes;
        auto nit = std::find(nodes.begin(), nodes.end(), failed_node_id);
        if (nit == nodes.end()) continue;

        // Remove the failed node
        nodes.erase(nit);

        if (nodes.empty()) {
            // Single-node job on the failed node → FAILED
            job.status = JobStatus::FAILED;
            job.completed_at_us = now_us();
            history_.push_back(job);
        } else {
            // Multi-node job → reschedule on remaining nodes
            job.status = JobStatus::RESCHEDULING;
            queue_.push_front(job);
        }
    }

    // Remove failed/evicted jobs from active list
    active_.erase(
        std::remove_if(active_.begin(), active_.end(),
                       [](const JobDescriptor& j) {
                           return j.status == JobStatus::FAILED ||
                                  j.status == JobStatus::RESCHEDULING;
                       }),
        active_.end());

    DISTRO_WARN("Scheduler: evicted node %u — affected %zu jobs",
                failed_node_id, history_.size());
}

// ── Internal helpers ───────────────────────────────────────────────────────

bool Scheduler::can_allocate(const JobDescriptor& job,
                              const std::vector<uint16_t>& online_nodes,
                              const std::vector<NodeLoad>& node_loads) const {
    int available = 0;
    for (size_t i = 0; i < online_nodes.size(); i++) {
        uint16_t nid = online_nodes[i];
        if (nid >= node_loads.size()) continue;

        const auto& load = node_loads[nid];
        if (load.memory_used + job.memory_per_node <= config_.memory_per_node) {
            available++;
        }
    }
    return available >= job.requested_nodes;
}

void Scheduler::allocate(JobDescriptor& job,
                          const std::vector<uint16_t>& online_nodes,
                          std::vector<NodeLoad>& node_loads) {
    job.assigned_nodes.clear();

    for (uint16_t nid : online_nodes) {
        if (static_cast<int>(job.assigned_nodes.size()) >= job.requested_nodes) break;
        if (nid >= node_loads.size()) continue;

        auto& load = node_loads[nid];
        if (load.memory_used + job.memory_per_node <= config_.memory_per_node) {
            job.assigned_nodes.push_back(nid);
            load.memory_used += job.memory_per_node;
            load.active_jobs++;
        }
    }
}

} // namespace distro
