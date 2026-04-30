#pragma once

#include <mutex>

namespace distro {

/// Serializes MCCL-owned Metal command-queue work, global staging blits, and
/// paired MPS stream drain + MCCL queue drain. Recursive so nested helpers
/// (e.g. metal_reduce_op -> metal_accumulate_chunk) do not self-deadlock.
/// Not held across PyTorch dispatch_sync (see EventSync).
std::recursive_mutex& distro_device_ops_mutex();

} // namespace distro
