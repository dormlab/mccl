#include "dmem/MemoryCatalog.hpp"
#include "common/Logging.hpp"

#include <cstring>
#include <algorithm>

namespace distro {

void MemoryCatalog::upsert(uint32_t region_id, const RegionDirectoryEntry& entry) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t key = catalog_key(0, region_id);  // node_id=0 for local entries
    entries_[key] = entry;
    version_.fetch_add(1);
}

void MemoryCatalog::remove(uint32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t key = catalog_key(0, region_id);
    entries_.erase(key);
    version_.fetch_add(1);
}

void MemoryCatalog::remove_node(uint16_t node_id) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t prefix = catalog_key(node_id, 0) & 0xFFFFFFFFFFFF0000ULL;
    auto it = entries_.begin();
    while (it != entries_.end()) {
        if ((it->first & 0xFFFFFFFFFFFF0000ULL) == prefix) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
    version_.fetch_add(1);
}

const RegionDirectoryEntry* MemoryCatalog::lookup(uint16_t node_id,
                                                    uint32_t region_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t key = catalog_key(node_id, region_id);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return &it->second;
    }
    return nullptr;
}

void MemoryCatalog::deserialize_from(const uint8_t* buf, size_t len,
                                      uint16_t source_node) {
    std::lock_guard<std::mutex> lock(mu_);

    size_t count = len / RegionDirectoryEntry::WIRE_SIZE;
    for (size_t i = 0; i < count; i++) {
        RegionDirectoryEntry entry =
            RegionDirectoryEntry::deserialize(buf + i * RegionDirectoryEntry::WIRE_SIZE);

        // region_id = i (implicit from position in catalog buffer)
        uint64_t key = catalog_key(source_node, static_cast<uint32_t>(i));
        entries_[key] = entry;
    }

    version_.fetch_add(1);
}

size_t MemoryCatalog::serialize_into(uint8_t* buf, size_t max_len) const {
    std::lock_guard<std::mutex> lock(mu_);

    size_t written = 0;
    for (const auto& [key, entry] : entries_) {
        // Only serialize local entries (node_id=0 in the key packing)
        uint16_t node = static_cast<uint16_t>(key >> 48);
        if (node != 0) continue;  // Skip cached remote entries

        uint32_t region_id = static_cast<uint32_t>((key >> 16) & 0xFFFFFFFF);
        if (region_id >= MAX_REGIONS_PER_NODE) continue;

        size_t offset = region_id * RegionDirectoryEntry::WIRE_SIZE;
        if (offset + RegionDirectoryEntry::WIRE_SIZE > max_len) break;

        entry.serialize(buf + offset);
        written = std::max(written, offset + RegionDirectoryEntry::WIRE_SIZE);
    }

    return written;
}

size_t MemoryCatalog::local_entry_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t count = 0;
    for (const auto& [key, _] : entries_) {
        uint16_t node = static_cast<uint16_t>(key >> 48);
        if (node == 0) count++;
    }
    return count;
}

} // namespace distro
