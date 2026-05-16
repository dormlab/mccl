#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace distro {

void register_mccl_backend(py::module_& m);

PYBIND11_MODULE(_C, m) {
    m.doc() = "MCCL c10d backend";
    register_mccl_backend(m);
}

} // namespace distro
