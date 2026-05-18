#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <mutex>
#include <vector>

namespace mccl {

/// c10d::Work for MCCL collectives.
///
/// A WorkMCCL is created when a collective is issued. The progress engine
/// (or the issuing thread, for synchronous ops) calls `finish()` or
/// `finishWithException()` when the underlying transport/compute completes.
/// `wait()` blocks until then; `getFuture()` returns a fulfilled Future
/// once the work is done.
class WorkMCCL : public c10d::Work {
public:
    WorkMCCL(c10d::OpType op_type,
             std::vector<at::Tensor> output_tensors,
             const char* profiling_title = "mccl:work");

    ~WorkMCCL() override;

    bool isCompleted() override;
    bool isSuccess() const override;
    std::exception_ptr exception() const override;

    /// Block until the op is finished or timeout elapses. Returns true on success.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    /// Tensors produced by this op (for collectives that return tensors).
    std::vector<at::Tensor> result() override;

    /// Future that is satisfied with `output_tensors_` when this op completes.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    /// Mark the work as finished. Thread-safe; calling multiple times is a no-op.
    void finish();

    /// Mark the work as finished with an error. Thread-safe.
    void finishWithException(std::exception_ptr eptr);

    /// Set a function to run on the issuing thread when finish() fires.
    /// Used to synchronize Metal command buffers before signalling waiters.
    void setSyncCallback(std::function<void()> cb) { sync_cb_ = std::move(cb); }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> completed_{false};
    std::exception_ptr exception_;

    std::vector<at::Tensor> output_tensors_;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::function<void()> sync_cb_;
};

} // namespace mccl
