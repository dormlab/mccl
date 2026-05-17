#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace mccl {

/// Single-threaded FIFO queue of pending collective tasks.
///
/// Each ProcessGroup owns one of these. Collective methods enqueue a task
/// and return immediately with a WorkMCCL; the engine thread runs tasks
/// in submission order and fulfills the Work when done. FIFO order is
/// required for correctness: cross-rank send/recv frame matching depends
/// on every rank dispatching collectives in the same order.
class Progress {
public:
    Progress();
    ~Progress();

    void submit(std::function<void()> task);
    void shutdown();

private:
    void run();

    std::queue<std::function<void()>> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::thread t_;
};

} // namespace mccl
