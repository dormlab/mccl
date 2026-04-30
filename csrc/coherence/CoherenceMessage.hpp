#pragma once

#include "dmem/GlobalAddress.hpp"

#include <cstdint>

namespace distro {

/// Coherence protocol opcodes — delivered via RDMA write_with_imm.
///
/// Each message carries a 32-bit immediate value (the opcode) on the
/// receiver's CQ.  The message body (RDMA-written data) contains the
/// page address, epoch, and sender sequence number.

enum class CoherenceOp : uint32_t {
    // ── Reader → Home ──────────────────────────────────────────────────
    READ_REQ          = 0x3001,  // "I want to read this page"
    READ_REPLY        = 0x3002,  // "Here is the data" (home → reader)
    READ_REPLY_DIRTY  = 0x3003,  // "Data fetched from exclusive owner → reader"

    // ── Writer → Home ──────────────────────────────────────────────────
    EXCLUSIVE_REQ     = 0x4001,  // "I want exclusive (write) access"
    EXCLUSIVE_GRANT   = 0x4002,  // "Exclusive granted, no invalidations needed"
    EXCLUSIVE_DENY    = 0x4003,  // "Exclusive denied — sharers exist, invalidating"

    // ── Home → Sharer ──────────────────────────────────────────────────
    INVALIDATE        = 0x1001,  // "Discard your copy — another node is writing"
    INVAL_ACK         = 0x1002,  // "Invalidation complete"

    // ── Home → Exclusive owner ─────────────────────────────────────────
    WRITEBACK_REQ     = 0x2001,  // "Someone wants your dirty data — flush it"
    WRITEBACK_DATA    = 0x2002,  // "Here is the dirty data" (owner → home)
    WRITEBACK_ACK     = 0x2003,  // "Writeback + invalidation complete"

    // ── Global synchronization ─────────────────────────────────────────
    BARRIER_ENTER     = 0x5001,  // "I've reached the barrier"
    BARRIER_LEAVE     = 0x5002,  // "All nodes reached — proceed"
};

/// Coherence message body: identifies the page under negotiation.
struct CoherenceMessage {
    CoherenceOp op;
    GlobalAddress page_addr;  // Which page (node_id, region_id, page_offset)
    uint64_t epoch;           // Directory epoch at time of message
    uint64_t sender_seq;      // Sender's monotonic sequence for ordering
};

/// Coherence page size: 64 KB (16 Thunderbolt-5 MTUs).
static constexpr uint64_t COHERENCE_PAGE_SIZE = 64 * 1024;

/// Align an offset down to the nearest page boundary.
inline uint64_t page_align_down(uint64_t offset) {
    return offset & ~(COHERENCE_PAGE_SIZE - 1);
}

/// Compute which page a byte offset falls in.
inline uint64_t page_index(uint64_t offset) {
    return offset / COHERENCE_PAGE_SIZE;
}

/// Size of the coherence message buffer per peer (1 page).
static constexpr size_t COHERENCE_MSG_BUF_SIZE = COHERENCE_PAGE_SIZE;

} // namespace distro
