#include "backend/WorkMCCL.hpp"

#include <ATen/core/ivalue.h>
#include <c10/util/intrusive_ptr.h>

namespace distro {

namespace {

c10::intrusive_ptr<c10::ivalue::Future> make_future(
    const std::vector<at::Tensor>& tensors) {
    std::vector<c10::Device> devices;
    devices.reserve(tensors.size());
    for (const auto& t : tensors) {
        devices.emplace_back(t.device());
    }
    auto future = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    return future;
}

} // namespace

WorkMCCL::WorkMCCL(c10d::OpType op_type,
                   std::vector<at::Tensor> output_tensors,
                   const char* profiling_title)
    : c10d::Work(/*rank=*/-1, op_type, profiling_title, output_tensors),
      output_tensors_(std::move(output_tensors)) {
    future_ = make_future(output_tensors_);
}

WorkMCCL::~WorkMCCL() = default;

bool WorkMCCL::isCompleted() {
    return completed_.load(std::memory_order_acquire);
}

bool WorkMCCL::isSuccess() const {
    std::lock_guard<std::mutex> lk(mu_);
    return completed_.load(std::memory_order_acquire) && !exception_;
}

std::exception_ptr WorkMCCL::exception() const {
    std::lock_guard<std::mutex> lk(mu_);
    return exception_;
}

bool WorkMCCL::wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timeout == kNoTimeout) {
        cv_.wait(lk, [this] { return completed_.load(std::memory_order_acquire); });
    } else {
        if (!cv_.wait_for(lk, timeout, [this] {
                return completed_.load(std::memory_order_acquire);
            })) {
            return false;
        }
    }
    if (exception_) {
        std::rethrow_exception(exception_);
    }
    return true;
}

std::vector<at::Tensor> WorkMCCL::result() {
    return output_tensors_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkMCCL::getFuture() {
    return future_;
}

void WorkMCCL::finish() {
    if (sync_cb_) {
        try {
            sync_cb_();
        } catch (...) {
            finishWithException(std::current_exception());
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
    }
    if (future_ && !future_->completed()) {
        future_->markCompleted(c10::IValue(output_tensors_));
    }
    cv_.notify_all();
}

void WorkMCCL::finishWithException(std::exception_ptr eptr) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        exception_ = eptr;
    }
    if (future_ && !future_->completed()) {
        future_->setError(eptr);
    }
    cv_.notify_all();
}

} // namespace distro
