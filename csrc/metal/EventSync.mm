#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <torch/torch.h>
#import <torch/mps.h>
#import <ATen/mps/MPSStream.h>
#import <ATen/mps/MPSEvent.h>

#include "metal/EventSync.hpp"
#include "metal/MPSInterop.hpp"
#include "common/Errors.hpp"
#include "common/Logging.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace mccl {

namespace {

struct EventState {
    id<MTLSharedEvent> mps_event   = nil;
    id<MTLSharedEvent> mccl_event  = nil;
    id<MTLDevice>      device      = nil;
    id<MTLCommandQueue> mccl_queue = nil;
    std::atomic<uint64_t> counter{0};
    std::atomic<bool> initialized{false};
};

EventState& state() {
    static EventState s;
    return s;
}

void spin_wait_event(id<MTLSharedEvent> event, uint64_t target) {
    constexpr int FAST_SPINS = 200;
    constexpr int YIELD_SPINS = 2000;
    constexpr auto TIMEOUT = std::chrono::seconds(30);

    for (int i = 0; i < FAST_SPINS; ++i) {
        if (event.signaledValue >= target) return;
    }

    for (int i = 0; i < YIELD_SPINS; ++i) {
        std::this_thread::yield();
        if (event.signaledValue >= target) return;
    }

    auto deadline = std::chrono::steady_clock::now() + TIMEOUT;
    auto delay = std::chrono::microseconds(10);
    constexpr auto max_delay = std::chrono::microseconds(500);
    while (event.signaledValue < target) {
        std::this_thread::sleep_for(delay);
        if (delay < max_delay) {
            delay *= 2;
        }
        DISTRO_CHECK(std::chrono::steady_clock::now() < deadline,
                   "spin_wait_event timed out after 30s waiting for event value " +
                   std::to_string(target) + " (current=" +
                   std::to_string(event.signaledValue) + ")");
    }
}

} // anonymous namespace


void event_sync_init() {
    EventState& s = state();
    if (s.initialized.load(std::memory_order_acquire)) return;

    @autoreleasepool {
        s.device = (__bridge id<MTLDevice>)get_mtl_device();
        if (!s.device) {
            DISTRO_WARN("EventSync: no Metal device, event sync disabled");
            return;
        }

        s.mccl_queue = (__bridge id<MTLCommandQueue>)get_mccl_command_queue();

        s.mps_event = [s.device newSharedEvent];
        if (!s.mps_event) {
            DISTRO_WARN("EventSync: MTLSharedEvent creation failed (mps_event)");
            return;
        }

        s.mccl_event = [s.device newSharedEvent];
        if (!s.mccl_event) {
            DISTRO_WARN("EventSync: MTLSharedEvent creation failed (mccl_event)");
            s.mps_event = nil;
            return;
        }

        s.counter.store(0);
        s.initialized.store(true, std::memory_order_release);
        DISTRO_INFO("EventSync initialized (MTLSharedEvent-based sync)");
    }
}

bool event_sync_available() {
    return state().initialized.load(std::memory_order_acquire);
}

namespace {

// Pending events keyed by the sequence value returned from
// mps_event_sync_nonblocking. The calling thread records the event on
// torch's current MPS stream; the engine thread looks it up by value
// and waits.
std::mutex g_pending_mu;
std::unordered_map<uint64_t, at::mps::MPSEventPtr> g_pending;

}  // namespace

void commit_mps_and_signal(uint64_t value) {
    EventState& s = state();
    DISTRO_CHECK(s.initialized, "EventSync not initialized");

    // Use torch's at::mps::MPSEvent — the public cross-thread sync
    // primitive on the MPS backend. record(needsLock=true) takes the
    // stream's internal lock before touching the command buffer, so
    // this is safe to call from any thread, including DDP's reducer
    // hook on the autograd thread while MPSGraph is encoding ops on
    // the same stream. The engine thread waits on the event later
    // (wait_for_mps → synchronize), which blocks until torch commits
    // its current MPS buffer and the GPU finishes it.
    auto stream = at::mps::getCurrentMPSStream();
    auto event = at::mps::getMPSEventPool()->acquireEvent(/*enable_timing=*/false, stream);
    event->record(/*needsLock=*/true);
    std::lock_guard<std::mutex> lk(g_pending_mu);
    g_pending.emplace(value, std::move(event));
}

void wait_for_mps(uint64_t value) {
    at::mps::MPSEventPtr event;
    {
        std::lock_guard<std::mutex> lk(g_pending_mu);
        auto it = g_pending.find(value);
        if (it == g_pending.end()) return;
        event = std::move(it->second);
        g_pending.erase(it);
    }
    // Blocks until the recorded point on torch's MPS stream has been
    // executed on the GPU. Releases the engine thread to read CPU-side
    // bytes from the unified-memory buffer safely.
    event->synchronize();
}

void signal_mccl_done(uint64_t value) {
    EventState& s = state();
    DISTRO_CHECK(s.initialized, "EventSync not initialized");

    // For the CPU reduction path (f32 vDSP), results are written directly
    // to unified memory -- just update the event counter so anyone polling
    // knows we're done.  For the Metal shader path we encode the signal on
    // MCCL's command queue and commit.
    s.mccl_event.signaledValue = value;
}

void signal_mccl_done_gpu(uint64_t value) {
    EventState& s = state();
    DISTRO_CHECK(s.initialized, "EventSync not initialized");

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [s.mccl_queue commandBuffer];
        cmd.label = @"mccl_signal_done";
        [cmd encodeSignalEvent:s.mccl_event value:value];
        [cmd commit];
    }
}

void wait_for_mccl(uint64_t value) {
    EventState& s = state();
    DISTRO_CHECK(s.initialized, "EventSync not initialized");
    spin_wait_event(s.mccl_event, value);
}

uint64_t next_event_value() {
    return state().counter.fetch_add(1) + 1;
}

} // namespace mccl
