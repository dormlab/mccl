#pragma once

#include <cstdint>
#include <cstddef>

namespace distro {

/// Control protocol between head node and mini agents over TCP.
///
/// The head sends commands; agents respond.  All multi-byte fields are
/// little-endian.  This is intentionally NOT RDMA — the control plane is
/// low-bandwidth and benefits from TCP's simplicity.

enum class AgentCmd : uint16_t {
    PING          = 0x0001,  // Liveness check
    REGISTER      = 0x0010,  // Register a Metal buffer with RDMA NIC
    DEREGISTER    = 0x0011,  // Unregister a region
    SHADER_LOAD   = 0x0020,  // Load a metallib from a path
    SHADER_RUN    = 0x0021,  // Run a named kernel on a registered buffer
    SHADER_SYNC   = 0x0022,  // Block until all GPU work completes
    STATS         = 0x0030,  // Query memory/compute stats
    SHUTDOWN      = 0xFFFF,  // Graceful shutdown
};

enum class AgentStatus : uint16_t {
    OK            = 0x0000,
    ERR_NOT_FOUND = 0x0001,
    ERR_MEMORY    = 0x0002,
    ERR_METAL     = 0x0003,
    ERR_RDMA      = 0x0004,
    ERR_INTERNAL  = 0xFFFF,
};

/// Fixed-size command header (8 bytes).
struct AgentHeader {
    AgentCmd cmd;
    AgentStatus status;  // Zero on request, set in response
    uint32_t payload_len; // Bytes following the header

    static constexpr size_t WIRE_SIZE = 8;
};

/// REGISTER request: tell the agent to register a buffer region.
struct RegisterRequest {
    uint64_t buffer_addr;   // MTLBuffer.contents pointer
    uint64_t length;        // Bytes
    uint8_t  flags;         // MemoryRegion::Flags (READABLE|WRITABLE|SHARED)

    static constexpr size_t WIRE_SIZE = 8 + 8 + 1;
};

/// REGISTER response.
struct RegisterResponse {
    uint32_t region_id;     // Assigned region ID
    uint32_t rkey;          // Remote key for RDMA access

    static constexpr size_t WIRE_SIZE = 4 + 4;
};

/// SHADER_RUN request: run a Metal compute kernel.
struct ShaderRunRequest {
    char     kernel_name[64];   // Null-terminated Metal function name
    uint32_t region_id;         // Buffer region to operate on
    uint64_t element_count;     // Number of elements
    float    scale;             // Scale factor (for scale/accumulate_scale ops)
    uint8_t  op;                // 0=accumulate, 1=scale, 2=accumulate_scale,
                                // 3=min, 4=max, 5=product

    static constexpr size_t WIRE_SIZE = 64 + 4 + 8 + 4 + 1;
};

/// STATS response.
struct StatsResponse {
    uint64_t total_put_bytes;
    uint64_t total_get_bytes;
    uint64_t total_put_ops;
    uint64_t total_get_ops;
    uint32_t region_count;
    uint64_t free_memory;
    uint32_t gpu_temp_celsius;    // 0 if unavailable
    uint8_t  link_state;          // 0=down, 1=up, 2=degraded

    static constexpr size_t WIRE_SIZE = 8 + 8 + 8 + 8 + 4 + 8 + 4 + 1;
};

} // namespace distro
