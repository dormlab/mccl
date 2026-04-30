#include "runtime/DistributedMetalRuntime.hpp"
#include "agent/Protocol.hpp"
#include "common/Logging.hpp"
#include "common/Errors.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>

namespace distro {

// ── Construction / destruction ──────────────────────────────────────────────

DistributedMetalRuntime::DistributedMetalRuntime(const Config& config,
                                                   DistributedMemoryManager* dmem,
                                                   CoherenceProtocol* coherence)
    : node_id_(config.node_id)
    , is_head_(config.is_head)
    , dmem_(dmem)
    , coherence_(coherence)
{
    DISTRO_CHECK(dmem_ != nullptr, "DMR: DMEM is required");
    DISTRO_CHECK(coherence_ != nullptr, "DMR: CoherenceProtocol is required");

    // Parse agent host:port strings and connect
    if (is_head_) {
        for (size_t i = 0; i < config.agent_hosts.size(); i++) {
            const auto& hp = config.agent_hosts[i];
            auto colon = hp.rfind(':');
            std::string host = hp.substr(0, colon);
            uint16_t port = static_cast<uint16_t>(
                std::stoi(hp.substr(colon + 1)));

            int sock = connect_to_agent(host, port);
            if (sock >= 0) {
                uint16_t nid = static_cast<uint16_t>(i + 1);  // Agent IDs start at 1
                agent_socks_[nid] = sock;
                node_ids_.push_back(nid);
                DISTRO_INFO("DMR: connected to agent %u at %s:%u", nid,
                            host.c_str(), port);
            } else {
                DISTRO_WARN("DMR: could not connect to agent at %s:%u",
                            host.c_str(), port);
            }
        }
    }

    DISTRO_INFO("DMR: initialized (head=%s, agents=%zu)",
                is_head_ ? "yes" : "no", node_ids_.size());
}

DistributedMetalRuntime::~DistributedMetalRuntime() {
    for (auto& [nid, sock] : agent_socks_) {
        disconnect_agent(nid);
    }
}

// ── Library management ────────────────────────────────────────────────────

int DistributedMetalRuntime::load_library(const std::string& metallib_path) {
    // Read the metallib file
    std::ifstream file(metallib_path, std::ios::binary | std::ios::ate);
    DISTRO_CHECK(file.good(), "DMR: cannot open metallib: " + metallib_path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    LibraryEntry entry{};
    entry.path = metallib_path;
    entry.data.resize(size);
    file.read(reinterpret_cast<char*>(entry.data.data()), size);
    entry.handle = static_cast<int>(libraries_.size());

    libraries_.push_back(std::move(entry));

    DISTRO_INFO("DMR: loaded metallib %s (%lld bytes, handle=%d)",
                metallib_path.c_str(), (long long)size, entry.handle);

    // Push to all agents
    for (uint16_t nid : node_ids_) {
        push_library_to_agent(nid, entry.handle);
    }

    return entry.handle;
}

bool DistributedMetalRuntime::push_library_to_agent(uint16_t node_id,
                                                     int lib_handle) {
    if (lib_handle < 0 || lib_handle >= static_cast<int>(libraries_.size())) {
        return false;
    }

    const auto& lib = libraries_[lib_handle];
    auto it = agent_socks_.find(node_id);
    if (it == agent_socks_.end()) return false;

    // For now, agents load from a shared path.  The metallib is installed
    // at a known location by setup.py (next to _C.so).  We just tell the
    // agent the path via SHADER_LOAD command.
    //
    // Full inline push (sending raw bytes) is a future optimization.

    AgentHeader hdr{AgentCmd::SHADER_LOAD, AgentStatus::OK, 0};
    (void)lib;
    (void)it;

    // TODO: implement shader push via TCP SHADER_LOAD command
    return true;
}

// ── Data-parallel dispatch ────────────────────────────────────────────────

void DistributedMetalRuntime::dispatch_data_parallel(
    const ShaderDispatch& dispatch)
{
    DISTRO_INFO("DMR: data-parallel dispatch '%s' (%llu elements, %zu nodes)",
                dispatch.kernel_name.c_str(),
                (unsigned long long)dispatch.global_element_count,
                dispatch.node_work.size());

    // ── Pre-dispatch coherence: acquire remote input buffers ──────────
    uint64_t epoch = coherence_->global_epoch();
    for (const auto& nw : dispatch.node_work) {
        for (const auto& binding : nw.bindings) {
            if (binding.is_input && binding.buffer.node_id != nw.node_id) {
                // Read the remote page — this triggers coherence state changes
                coherence_->read_page(binding.buffer, false);
            }
        }
    }
    coherence_->barrier_all();

    // ── Dispatch to each agent ───────────────────────────────────────
    for (const auto& nw : dispatch.node_work) {
        auto it = agent_socks_.find(nw.node_id);
        if (it == agent_socks_.end()) {
            DISTRO_WARN("DMR: no connection to agent %u, skipping", nw.node_id);
            continue;
        }

        int sock = it->second;

        // Build SHADER_RUN command
        ShaderRunRequest req{};
        std::memset(&req, 0, sizeof(req));
        std::strncpy(req.kernel_name, dispatch.kernel_name.c_str(),
                     sizeof(req.kernel_name) - 1);
        req.element_count = nw.grid_size;
        req.op = 0;  // Default: accumulate (may differ per binding)

        AgentHeader hdr{AgentCmd::SHADER_RUN, AgentStatus::OK,
                        ShaderRunRequest::WIRE_SIZE};

        // Send header + request
        ::send(sock, &hdr, sizeof(hdr), 0);
        ::send(sock, &req, sizeof(req), 0);
    }

    // ── Wait for completions ─────────────────────────────────────────
    for (const auto& nw : dispatch.node_work) {
        auto it = agent_socks_.find(nw.node_id);
        if (it == agent_socks_.end()) continue;

        int sock = it->second;
        AgentHeader resp{};
        ssize_t n = ::recv(sock, &resp, sizeof(resp), 0);
        if (n == sizeof(resp) && resp.status == AgentStatus::OK) {
            DISTRO_DEBUG("DMR: agent %u completed '%s'", nw.node_id,
                         dispatch.kernel_name.c_str());
        } else {
            DISTRO_ERROR("DMR: agent %u failed '%s' (status=%u)",
                         nw.node_id, dispatch.kernel_name.c_str(),
                         static_cast<uint16_t>(resp.status));
        }
    }

    // ── Post-dispatch coherence: release output buffers ──────────────
    for (const auto& nw : dispatch.node_work) {
        for (const auto& binding : nw.bindings) {
            if (binding.is_output && binding.buffer.node_id != nw.node_id) {
                coherence_->write_page_done(binding.buffer);
            }
        }
    }

    coherence_->barrier_all();
}

std::vector<std::future<void>>
DistributedMetalRuntime::dispatch_data_parallel_async(
    const ShaderDispatch& dispatch)
{
    (void)dispatch;
    // Async dispatch returns immediately; completion via future.
    // Each future is signaled when the agent responds.
    // Not yet implemented — uses sync dispatch for now.
    std::vector<std::future<void>> futures;
    return futures;
}

// ── Pipeline-parallel dispatch ────────────────────────────────────────────

void DistributedMetalRuntime::dispatch_pipeline(
    const std::vector<PipelineStage>& stages,
    int num_micro_batches)
{
    DISTRO_INFO("DMR: pipeline dispatch (%zu stages, %d micro-batches)",
                stages.size(), num_micro_batches);

    int num_stages = static_cast<int>(stages.size());

    // ── Warm-up: fill the pipeline ────────────────────────────────────
    for (int mb = 0; mb < std::min(num_micro_batches, num_stages); mb++) {
        // Each active stage processes one micro-batch
        for (int s = 0; s <= mb && s < num_stages; s++) {
            const auto& stage = stages[s];

            ShaderRunRequest req{};
            std::strncpy(req.kernel_name, stage.kernel_name.c_str(),
                         sizeof(req.kernel_name) - 1);
            req.element_count = stage.forward_buffer_size / sizeof(float);
            req.op = 0;

            AgentHeader hdr{AgentCmd::SHADER_RUN, AgentStatus::OK,
                            ShaderRunRequest::WIRE_SIZE};

            auto it = agent_socks_.find(stage.node_id);
            if (it != agent_socks_.end()) {
                ::send(it->second, &hdr, sizeof(hdr), 0);
                ::send(it->second, &req, sizeof(req), 0);
            }

            // RDMA transfer to next stage (if not last stage)
            if (s < num_stages - 1) {
                const auto& next = stages[s + 1];
                transfer_tensor(stage.forward_buffer, next.node_id,
                                next.forward_buffer,
                                stage.forward_buffer_size);
            }
        }
    }

    // ── Steady state: all stages active ──────────────────────────────
    for (int mb = num_stages; mb < num_micro_batches; mb++) {
        for (int s = 0; s < num_stages; s++) {
            const auto& stage = stages[s];

            // Wait for previous stage's RDMA transfer (if not first)
            if (s > 0) {
                dmem_->drain_pending();
            }

            // Dispatch this stage
            ShaderRunRequest req{};
            std::strncpy(req.kernel_name, stage.kernel_name.c_str(),
                         sizeof(req.kernel_name) - 1);
            req.element_count = stage.forward_buffer_size / sizeof(float);

            AgentHeader hdr{AgentCmd::SHADER_RUN, AgentStatus::OK,
                            ShaderRunRequest::WIRE_SIZE};

            // Toggle double buffer slot
            stage.use_slot_a = !stage.use_slot_a;

            auto it = agent_socks_.find(stage.node_id);
            if (it != agent_socks_.end()) {
                ::send(it->second, &hdr, sizeof(hdr), 0);
                ::send(it->second, &req, sizeof(req), 0);
            }

            // RDMA transfer to next stage
            if (s < num_stages - 1) {
                const auto& next = stages[s + 1];
                transfer_tensor(stage.forward_buffer, next.node_id,
                                next.forward_buffer,
                                stage.forward_buffer_size);
            }
        }
    }

    // ── Cool-down: drain the pipeline ────────────────────────────────
    for (int mb = num_micro_batches; mb < num_micro_batches + num_stages - 1; mb++) {
        for (int s = mb - num_micro_batches + 1; s < num_stages; s++) {
            if (s < 0) continue;

            const auto& stage = stages[s];
            ShaderRunRequest req{};
            std::strncpy(req.kernel_name, stage.kernel_name.c_str(),
                         sizeof(req.kernel_name) - 1);

            AgentHeader hdr{AgentCmd::SHADER_RUN, AgentStatus::OK,
                            ShaderRunRequest::WIRE_SIZE};

            auto it = agent_socks_.find(stage.node_id);
            if (it != agent_socks_.end()) {
                ::send(it->second, &hdr, sizeof(hdr), 0);
                ::send(it->second, &req, sizeof(req), 0);
            }
        }
    }

    // ── Read all responses ───────────────────────────────────────────
    for (int i = 0; i < num_micro_batches * num_stages; i++) {
        // Each dispatch produced a response
        for (const auto& stage : stages) {
            auto it = agent_socks_.find(stage.node_id);
            if (it != agent_socks_.end()) {
                AgentHeader resp{};
                ::recv(it->second, &resp, sizeof(resp), 0);
            }
        }
    }

    sync_all();
}

// ── Transfers ─────────────────────────────────────────────────────────────

std::future<void> DistributedMetalRuntime::transfer_tensor(
    GlobalAddress src, uint16_t dst_node,
    GlobalAddress dst, uint64_t length)
{
    // Issue an RDMA get: destination node pulls from source node.
    // The destination node initiates the RDMA read.
    uint64_t wr_id = dmem_->get(src.node_id, src, nullptr, length);

    // Wrap in a promise/future for async completion
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    // Poll for completion in a detached thread (production would use
    // a dedicated transfer thread pool)
    std::thread([this, wr_id, promise]() {
        bool ok = dmem_->poll_completion(wr_id);
        if (ok) {
            promise->set_value();
        } else {
            try {
                throw std::runtime_error("RDMA transfer failed");
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }
    }).detach();

    return future;
}

// ── Synchronization ───────────────────────────────────────────────────────

void DistributedMetalRuntime::sync_all() {
    // Tell all agents to sync their GPU command queues
    for (auto& [nid, sock] : agent_socks_) {
        AgentHeader hdr{AgentCmd::SHADER_SYNC, AgentStatus::OK, 0};
        ::send(sock, &hdr, sizeof(hdr), 0);

        AgentHeader resp{};
        ::recv(sock, &resp, sizeof(resp), 0);
    }

    // Cluster-wide coherence barrier
    coherence_->barrier_all();

    DISTRO_DEBUG("DMR: sync_all complete");
}

// ── Agent TCP client ──────────────────────────────────────────────────────

int DistributedMetalRuntime::connect_to_agent(const std::string& host,
                                               uint16_t port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        return -1;
    }

    // Non-blocking connect with 5-second timeout
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    int rc = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
        close(sock);
        return -1;
    }

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(sock, &fdset);
    struct timeval tv{5, 0};  // 5 seconds

    rc = select(sock + 1, nullptr, &fdset, nullptr, &tv);
    if (rc <= 0) {
        close(sock);
        return -1;
    }

    fcntl(sock, F_SETFL, flags);  // Restore blocking mode

    int opt = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    return sock;
}

void DistributedMetalRuntime::disconnect_agent(uint16_t node_id) {
    auto it = agent_socks_.find(node_id);
    if (it != agent_socks_.end()) {
        // Send SHUTDOWN command
        AgentHeader hdr{AgentCmd::SHUTDOWN, AgentStatus::OK, 0};
        ::send(it->second, &hdr, sizeof(hdr), 0);
        close(it->second);
        agent_socks_.erase(it);
    }
}

} // namespace distro
