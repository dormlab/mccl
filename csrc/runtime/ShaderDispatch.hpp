#pragma once

#include "dmem/GlobalAddress.hpp"

#include <cstdint>
#include <vector>
#include <string>

namespace distro {

/// Describes how a compute shader is partitioned across nodes.
///
/// The head compiles one metallib and pushes it to all agents.  Each agent
/// receives a shader name and the subset of the global grid it should run.
///
/// Grid partitioning is 1D (along x) by default; the global grid size is
/// split evenly across nodes.  Input and output buffers are described by
/// their GlobalAddress so the coherence layer can ensure visibility.

struct ShaderBinding {
    uint32_t      index;        // Buffer binding index (0..N)
    GlobalAddress buffer;       // Global address of the buffer region
    uint64_t      size;         // Bytes to bind
    bool          is_input;     // True if shader reads this buffer
    bool          is_output;    // True if shader writes this buffer
};

/// Per-node work partition for a data-parallel dispatch.
struct NodeWork {
    uint16_t node_id;
    uint64_t grid_offset;       // Offset into the global 1D grid
    uint64_t grid_size;         // Number of elements this node processes
    uint32_t threadgroup_count; // Computed from grid_size / (threads_per_group * 8)

    std::vector<ShaderBinding> bindings;  // Buffers for this node's shard
};

/// Fully-described data-parallel shader dispatch.
struct ShaderDispatch {
    std::string kernel_name;       // Metal function name (e.g., "accumulate_chunk_f32")
    std::string metallib_path;     // Path to the precompiled metallib on each agent

    uint64_t global_element_count; // Total elements across all nodes
    uint32_t threadgroup_size;     // Threads per threadgroup (from pipeline state)
    uint32_t elements_per_thread;  // 8 for vectorized kernels

    std::vector<NodeWork> node_work;  // One entry per participating node

    /// Compute a 1D partition across nodes.
    static ShaderDispatch data_parallel(
        const std::string& kernel_name,
        const std::string& metallib_path,
        uint64_t total_elements,
        uint32_t threadgroup_size,
        uint32_t elements_per_thread,
        const std::vector<uint16_t>& node_ids);
};

/// A stage in a pipeline-parallel dispatch.
///
/// Each stage runs on a specific node.  Data flows from stage to stage
/// through double-buffered RDMA regions.
struct PipelineStage {
    uint16_t node_id;
    std::string kernel_name;
    std::vector<ShaderBinding> inputs;
    std::vector<ShaderBinding> outputs;

    /// RDMA buffer for passing data to the next stage.
    /// Stage N writes here, Stage N+1 reads from here.
    GlobalAddress forward_buffer;
    uint64_t    forward_buffer_size;

    /// Double-buffer: two slots so writes and reads overlap.
    bool use_slot_a = true;
};

/// Metadata for a compiled Metal library distributed to agents.
struct DistributedLibrary {
    std::string path;               // Local path on the head
    std::string remote_path;        // Path where agents can find it
    std::vector<uint8_t> data;      // Raw metallib bytes (for inline push)
    bool loaded_on_agents = false;
};

// ── ShaderDispatch inline implementation ─────────────────────────────────

inline ShaderDispatch ShaderDispatch::data_parallel(
    const std::string& kernel_name,
    const std::string& metallib_path,
    uint64_t total_elements,
    uint32_t threadgroup_size,
    uint32_t elements_per_thread,
    const std::vector<uint16_t>& node_ids)
{
    ShaderDispatch sd{};
    sd.kernel_name          = kernel_name;
    sd.metallib_path        = metallib_path;
    sd.global_element_count = total_elements;
    sd.threadgroup_size     = threadgroup_size;
    sd.elements_per_thread  = elements_per_thread;

    int num_nodes = static_cast<int>(node_ids.size());
    uint64_t elements_per_node = total_elements / num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        NodeWork nw{};
        nw.node_id     = node_ids[i];
        nw.grid_offset = i * elements_per_node;
        nw.grid_size   = (i == num_nodes - 1)
                            ? total_elements - nw.grid_offset
                            : elements_per_node;

        uint64_t threads = (nw.grid_size + elements_per_thread - 1) / elements_per_thread;
        nw.threadgroup_count = static_cast<uint32_t>(
            (threads + threadgroup_size - 1) / threadgroup_size);

        sd.node_work.push_back(std::move(nw));
    }

    return sd;
}

} // namespace distro
