#include "backend/ProcessGroupMCCL.hpp"

#include <torch/extension.h>
#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>


#include <chrono>

namespace py = pybind11;

namespace mccl {

/// Called by `mccl._C._register_mccl_backend(module)` so the pybind11
/// module that owns the symbols is the same one Python imports.
///
/// Exposes:
///   - `ProcessGroupMCCL` class (so torch.distributed can introspect it)
///   - `_create_process_group_mccl(store, rank, size, timeout)` factory
///     used by `torch.distributed.Backend.register_backend("mccl", ...)`.
void register_mccl_backend(py::module_& m) {
    m.def("_create_process_group_mccl",
          [](const c10::intrusive_ptr<c10d::Store>& store,
             int rank, int size,
             std::chrono::milliseconds timeout)
              -> c10::intrusive_ptr<c10d::Backend> {
              ProcessGroupMCCL::Options opts;
              opts.timeout = timeout;
              return c10::make_intrusive<ProcessGroupMCCL>(
                  store, rank, size, opts);
          },
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(30 * 60 * 1000),
          "Factory used by torch.distributed.Backend.register_backend('mccl').");
}

} // namespace mccl
