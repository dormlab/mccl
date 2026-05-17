#include "coherence/CoherenceDirectory.hpp"
#include "common/Logging.hpp"

namespace mccl {

DirectoryEntry& CoherenceDirectory::get_or_create(uint32_t region_id,
                                                    uint64_t page_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t key = (static_cast<uint64_t>(region_id) << 32) | (page_idx & 0xFFFFFFFF);
    return entries_[key];
}

const DirectoryEntry* CoherenceDirectory::find(uint32_t region_id,
                                                 uint64_t page_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t key = (static_cast<uint64_t>(region_id) << 32) | (page_idx & 0xFFFFFFFF);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return &it->second;
    }
    return nullptr;
}

void CoherenceDirectory::remove_region(uint32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t prefix = static_cast<uint64_t>(region_id) << 32;
    auto it = entries_.begin();
    while (it != entries_.end()) {
        if ((it->first & 0xFFFFFFFF00000000ULL) == prefix) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
    DISTRO_DEBUG("CoherenceDirectory: removed region %u entries", region_id);
}

size_t CoherenceDirectory::page_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

} // namespace mccl
