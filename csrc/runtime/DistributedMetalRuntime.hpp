#pragma once

#include "runtime/ShaderDispatch.hpp"
#include "dmem/DistributedMemoryManager.hpp"
#include "coherence/CoherenceProtocol.hpp"

#include <memory>
#include <vector>
#include <string>
#include <future>
#include <unordered_map>

namespace mccl {

/// Distributed Metal Runtime — presents a cluster as a single logical GPU.
///
/// The head node (this runtime) partitions shader work across agent nodes.
/// Agents receive shader binaries and grid partitions over TCP, execute
/// Metal compute on their local MPS GPU, and report completion.
///
/// For data-parallel work, each node runs the same kernel on its slice of
/// the global grid.  Coherence acquire/release ensures remote buffer
/// visibility before and after shader execution.
///
/// For pipeline-parallel work, nodes form a pipeline with RDMA transfers
/// between stages.  Double-buffered regions allow transfer/compute overlap.

class DistributedMetalRuntime {
public:
    struct Config {
        uint16_t node_id = 0;
        bool     is_head = true;
        std::vector<std::string> agent_hosts;    // "host:port" for each agent
        std::string metallib_search_path;        // Where to find/compile shaders
    };

    DistributedMetalRuntime(const Config& config,
                            DistributedMemoryManager* dmem,
                            CoherenceProtocol* coherence);
    ~DistributedMetalRuntime();

    DistributedMetalRuntime(const DistributedMetalRuntime&) = delete;
    DistributedMetalRuntime& operator=(const DistributedMetalRuntime&) = delete;

    // ── Library management ──────────────────────────────────────────

    /// Compile a metallib from Metal source and distribute to all agents.
    /// Returns a library handle (index into libraries_ vector).
    int load_library(const std::string& metallib_path);

    /// Push raw metallib bytes to an agent.
    bool push_library_to_agent(uint16_t node_id, int lib_handle);

    // ── Data-parallel dispatch ──────────────────────────────────────

    /// Partition a shader across nodes and dispatch.
    /// Blocks until all nodes complete.
    void dispatch_data_parallel(const ShaderDispatch& dispatch);

    /// Non-blocking version: returns futures for per-node completion.
    std::vector<std::future<void>> dispatch_data_parallel_async(
        const ShaderDispatch& dispatch);

    // ── Pipeline-parallel dispatch ──────────────────────────────────

    /// Launch a pipeline across nodes with the given number of micro-batches.
    /// Warm-up → steady-state → cool-down.
    void dispatch_pipeline(const std::vector<PipelineStage>& stages,
                           int num_micro_batches);

    // ── Transfers ───────────────────────────────────────────────────

    /// RDMA transfer tensor data from one node to another.
    /// Non-blocking: returns a future for completion.
    std::future<void> transfer_tensor(GlobalAddress src, uint16_t dst_node,
                                       GlobalAddress dst, uint64_t length);

    // ── Synchronization ─────────────────────────────────────────────

    /// Wait for all pending GPU work to complete on all agents.
    void sync_all();

    // ── Agent communication ─────────────────────────────────────────

    /// Send a raw command to an agent. Returns true on success.
    bool send_agent_cmd(uint16_t node_id, uint16_t cmd,
                        const void* payload, uint32_t payload_len,
                        void* response, uint32_t* resp_len);

private:
    // ── Agent TCP client ────────────────────────────────────────────
    int connect_to_agent(const std::string& host, uint16_t port);
    void disconnect_agent(uint16_t node_id);

    // ── Shader library tracking ─────────────────────────────────────
    struct LibraryEntry {
        std::string path;
        std::vector<uint8_t> data;    // Raw metallib bytes
        int handle;
    };
    std::vector<LibraryEntry> libraries_;

    // ── Membership ──────────────────────────────────────────────────
    uint16_t node_id_;
    bool     is_head_;
    DistributedMemoryManager* dmem_;
    CoherenceProtocol* coherence_;

    // Agent TCP sockets (indexed by node_id)
    std::unordered_map<uint16_t, int> agent_socks_;

    // Known node IDs
    std::vector<uint16_t> node_ids_;
};

} // namespace mccl
