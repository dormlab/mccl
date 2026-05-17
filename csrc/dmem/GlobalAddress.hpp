#pragma once

#include <cstdint>
#include <cstring>

namespace mccl {

/// Global address: uniquely identifies a byte range in the cluster.
///
/// Wire format: 14 bytes packed — node_id (2B) | region_id (4B) | offset (8B).
struct GlobalAddress {
    uint16_t node_id    = 0;  // 0..MAX_NODES-1
    uint32_t region_id  = 0;  // Index into node's region table
    uint64_t offset     = 0;  // Byte offset within region

    static constexpr size_t WIRE_SIZE = 2 + 4 + 8;

    void serialize(uint8_t* buf) const {
        std::memcpy(buf,       &node_id,   2);
        std::memcpy(buf + 2,   &region_id, 4);
        std::memcpy(buf + 6,   &offset,    8);
    }

    static GlobalAddress deserialize(const uint8_t* buf) {
        GlobalAddress ga{};
        std::memcpy(&ga.node_id,   buf,       2);
        std::memcpy(&ga.region_id, buf + 2,   4);
        std::memcpy(&ga.offset,    buf + 6,   8);
        return ga;
    }

    /// Pack into a single 64-bit key for use as a hash key.
    uint64_t pack() const {
        return (static_cast<uint64_t>(node_id)   << 48) |
               (static_cast<uint64_t>(region_id) << 16);
    }

    bool operator==(const GlobalAddress& other) const {
        return node_id   == other.node_id   &&
               region_id == other.region_id &&
               offset    == other.offset;
    }

    bool operator!=(const GlobalAddress& other) const { return !(*this == other); }
};

/// Descriptor for a memory region registered with the local RDMA NIC.
///
/// Each region corresponds to an MTLBuffer (or portion thereof) that has been
/// registered via ibv_reg_mr.  On Apple Silicon with unified memory,
/// MTLBuffer.contents is a valid CPU pointer, so registration is zero-copy:
/// the NIC DMA engine reads/writes directly to GPU-visible memory.
struct MemoryRegion {
    GlobalAddress gaddr;       // How peers address this region
    uint64_t local_addr;       // Base virtual address (= buffer.contents + offset)
    uint64_t length;           // Size in bytes
    uint32_t rkey;             // Remote key (for RDMA READ/WRITE from peers)
    uint32_t lkey;             // Local key (for constructing local SGEs)

    enum Flags : uint8_t {
        READABLE   = 0x01,
        WRITABLE   = 0x02,
        SHARED     = 0x04,     // Coherence-managed (PGAS)
        PRIVATE    = 0x08,     // Not remotely accessible
    };
    uint8_t flags = 0;

    /// True if remote peers can RDMA-read this region.
    bool is_remote_readable() const { return flags & READABLE; }

    /// True if remote peers can RDMA-write this region.
    bool is_remote_writable() const { return flags & WRITABLE; }
};

/// Result of address resolution: either a local pointer (if this node
/// owns the memory) or remote RDMA parameters.
struct AddrResolution {
    bool     is_local     = false;
    void*    local_ptr    = nullptr;   // Valid if is_local
    uint64_t remote_addr  = 0;         // Remote VA for RDMA (if !is_local)
    uint32_t remote_rkey  = 0;         // Remote key (if !is_local)
    uint32_t local_lkey   = 0;         // Local MR lkey for SGE construction
    uint64_t length       = 0;         // Region length for bounds checking
};

} // namespace mccl
