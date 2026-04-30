#pragma once

#include "dmem/GlobalAddress.hpp"

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace distro {

/// Per-node region directory entry (published via well-known RDMA buffer).
///
/// Each node maintains a small catalog buffer that peers can RDMA-read to
/// discover available memory regions and their RDMA credentials.
struct RegionDirectoryEntry {
    uint64_t local_addr;     // Page-aligned base VA on the owning node
    uint64_t length;         // Size in bytes
    uint32_t rkey;           // Remote key for RDMA READ/WRITE to this region
    uint32_t flags;          // MemoryRegion::Flags
    uint64_t epoch;          // Bumped on each re-registration (invalidation signal)

    static constexpr size_t WIRE_SIZE = 8 + 8 + 4 + 4 + 8;  // 32 bytes

    void serialize(uint8_t* buf) const {
        std::memcpy(buf,       &local_addr, 8);
        std::memcpy(buf + 8,   &length,     8);
        std::memcpy(buf + 16,  &rkey,       4);
        std::memcpy(buf + 20,  &flags,      4);
        std::memcpy(buf + 24,  &epoch,      8);
    }

    static RegionDirectoryEntry deserialize(const uint8_t* buf) {
        RegionDirectoryEntry e{};
        std::memcpy(&e.local_addr, buf,       8);
        std::memcpy(&e.length,     buf + 8,   8);
        std::memcpy(&e.rkey,       buf + 16,  4);
        std::memcpy(&e.flags,      buf + 20,  4);
        std::memcpy(&e.epoch,      buf + 24,  8);
        return e;
    }
};

/// Cluster-wide memory catalog: replicated directory of all regions.
///
/// Each node's catalog is stored in a small RDMA-registered buffer at a
/// well-known GlobalAddress (region_id=0).  Peers RDMA-read this buffer to
/// discover and verify remote regions before issuing put()/get().
///
/// The local catalog (this node's own regions) is writable; remote catalogs
/// are cached copies refreshed via RDMA read on cache miss or epoch change.
class MemoryCatalog {
public:
    static constexpr size_t MAX_REGIONS_PER_NODE = 256;
    static constexpr size_t CATALOG_BUF_SIZE =
        MAX_REGIONS_PER_NODE * RegionDirectoryEntry::WIRE_SIZE;
    static constexpr uint32_t CATALOG_REGION_ID = 0;  // Well-known region ID

    /// Add or update an entry in the local catalog (this node's regions).
    void upsert(uint32_t region_id, const RegionDirectoryEntry& entry);

    /// Remove an entry from the local catalog.
    void remove(uint32_t region_id);

    /// Remove all catalog entries for a given node (used on node failure).
    void remove_node(uint16_t node_id);

    /// Look up an entry (local or cached remote).
    const RegionDirectoryEntry* lookup(uint16_t node_id, uint32_t region_id) const;

    /// Deserialize from a raw buffer (received via RDMA read from a peer).
    void deserialize_from(const uint8_t* buf, size_t len, uint16_t source_node);

    /// Serialize the local catalog into buf (for RDMA read by peers).
    size_t serialize_into(uint8_t* buf, size_t max_len) const;

    /// Current version — bumped on any local catalog change.
    uint64_t version() const { return version_.load(); }

    /// Number of entries in the local catalog.
    size_t local_entry_count() const;

private:
    // key = pack(node_id, region_id) — see GlobalAddress::pack()
    std::unordered_map<uint64_t, RegionDirectoryEntry> entries_;
    mutable std::mutex mu_;
    std::atomic<uint64_t> version_{0};
};

inline uint64_t catalog_key(uint16_t node_id, uint32_t region_id) {
    return (static_cast<uint64_t>(node_id)   << 48) |
           (static_cast<uint64_t>(region_id) << 16);
}

} // namespace distro
