#include "_test.h"
#include "backend/Progress.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("FIFO order across many tasks") {
    mccl::Progress p;
    std::mutex mu;
    std::vector<int> order;
    for (int i = 0; i < 100; ++i) {
        p.submit([i, &order, &mu] {
            std::lock_guard<std::mutex> lk(mu);
            order.push_back(i);
        });
    }
    p.shutdown();
    CHECK(order.size() == 100);
    for (int i = 0; i < 100; ++i) CHECK(order[i] == i);
}

TEST_CASE("dtor drains the queue") {
    std::atomic<int> count{0};
    {
        mccl::Progress p;
        for (int i = 0; i < 50; ++i) p.submit([&count] { ++count; });
    }
    CHECK(count.load() == 50);
}

TEST_CASE("shutdown is idempotent") {
    mccl::Progress p;
    p.shutdown();
    p.shutdown();
    CHECK(true);
}

TEST_CASE("exception in task does not poison the engine") {
    mccl::Progress p;
    std::atomic<int> ran{0};
    p.submit([] { throw std::runtime_error("boom"); });
    p.submit([&ran] { ++ran; });
    p.submit([&ran] { ++ran; });
    p.shutdown();
    CHECK(ran.load() == 2);
}

TEST_CASE("submit blocks for less than 1ms (lock-light)") {
    mccl::Progress p;
    // Long-running first task occupies the engine thread
    p.submit([] { std::this_thread::sleep_for(50ms); });
    auto t0 = std::chrono::steady_clock::now();
    p.submit([] {});
    auto t1 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    CHECK(us < 1000);
    p.shutdown();
}

MCCL_RUN_ALL()
