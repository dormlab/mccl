#pragma once

#include "coherence/CoherenceMessage.hpp"

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <bitset>

#include "dmem/DistributedMemoryManager.hpp"

namespace mccl {

/// Per-page coherence state tracked at the home node.
enum class CoherenceState : uint8_t {
    UNCACHED  = 0,   // No remote copy exists; home has authoritative data
    SHARED    = 1,   // One or more remote nodes hold clean read copies
    EXCLUSIVE = 2,   // Exactly one remote node holds the writable copy
};

/// Directory entry for a single 64 KB page.
///
/// The home node maintains one entry per page of registered memory.
/// All state transitions are serialized by the directory mutex.
struct DirectoryEntry {
    CoherenceState state = CoherenceState::UNCACHED;
    uint8_t sharers = 0;           // Bitmask: bit N = 1 → node N has a copy
    uint8_t exclusive_owner = 0;   // Valid only when state == EXCLUSIVE
    uint64_t epoch = 0;            // Monotonically increasing per-page version

    bool has_sharer(uint16_t node_id) const {
        return (sharers & (1 << node_id)) != 0;
    }

    void add_sharer(uint16_t node_id) {
        sharers |= (1 << node_id);
    }

    void remove_sharer(uint16_t node_id) {
        sharers &= ~(1 << node_id);
    }

    int sharer_count() const {
        return __builtin_popcount(sharers);
    }

    void clear_all_sharers() {
        sharers = 0;
        exclusive_owner = 0;
    }
};

/// Per-region coherence directory.
///
/// Indexed by page index (byte_offset / 64KB).  Only pages in regions
/// where this node is the home node have valid entries.
class CoherenceDirectory {
public:
    /// Look up or create an entry for a page.  Returns a reference valid
    /// until the next directory mutation (caller must hold the lock).
    DirectoryEntry& get_or_create(uint32_t region_id, uint64_t page_idx);

    /// Look up an entry. Returns nullptr if not found.
    const DirectoryEntry* find(uint32_t region_id, uint64_t page_idx) const;

    /// Remove all entries for a region (called on region unregister).
    void remove_region(uint32_t region_id);

    /// Number of tracked pages.
    size_t page_count() const;

private:
    // Key: (region_id << 32) | page_idx
    std::unordered_map<uint64_t, DirectoryEntry> entries_;
    mutable std::mutex mu_;
};

} // namespace mccl
