#include "backend/Progress.hpp"

namespace mccl {

Progress::Progress() : t_(&Progress::run, this) {}

Progress::~Progress() { shutdown(); }

void Progress::shutdown() {
    if (stop_.exchange(true)) return;
    cv_.notify_all();
    if (t_.joinable()) t_.join();
}

void Progress::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(std::move(task));
    }
    cv_.notify_one();
}

void Progress::run() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_.load() || !q_.empty(); });
            if (q_.empty()) {
                if (stop_.load()) return;
                continue;
            }
            task = std::move(q_.front());
            q_.pop();
        }
        try {
            task();
        } catch (...) {
            // task is responsible for surfacing exceptions via its WorkMCCL
        }
    }
}

} // namespace mccl
