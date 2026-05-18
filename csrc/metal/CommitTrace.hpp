#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#include <cstdio>
#include <atomic>

namespace mccl {

// Wraps [cmd commit] with a pre-commit sanity check. If the buffer is
// not in NotEnqueued state when we try to commit, log site + cmd ptr +
// status to stderr BEFORE Metal aborts so we can see which call site
// double-committed. The fflush ensures the message is visible even
// when the process is about to abort.
inline void mccl_commit_traced(id<MTLCommandBuffer> cmd, const char* site) {
    static std::atomic<uint64_t> ctr{0};
    uint64_t n = ctr.fetch_add(1, std::memory_order_relaxed);
    MTLCommandBufferStatus s = cmd.status;
    if (s != MTLCommandBufferStatusNotEnqueued) {
        fprintf(stderr,
                "[MCCL DOUBLE-COMMIT #%llu] site=%s cmd=%p status=%d\n",
                (unsigned long long)n, site, (__bridge void*)cmd, (int)s);
        fflush(stderr);
    }
    [cmd commit];
}

} // namespace mccl

#define MCCL_COMMIT(cmd, site) ::mccl::mccl_commit_traced((cmd), (site))

#endif // __OBJC__
