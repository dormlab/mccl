#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmem/DistributedMemoryManager.hpp"
#include "common/Version.hpp"

namespace py = pybind11;

namespace distro {

namespace {

// Singleton DMEM instance for the process (one per node)
std::unique_ptr<DistributedMemoryManager> g_dmem;

DistributedMemoryManager* get_dmem() {
    if (!g_dmem) {
        throw std::runtime_error("DMEM not initialized — call distro.init_dmem() first");
    }
    return g_dmem.get();
}

} // anonymous namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "MCCL Distributed Metal GPU Runtime";

    // ── Version ────────────────────────────────────────────────────────
    m.attr("__version__") = MCCL_VERSION_STRING;

    // ── DMEM initialization ────────────────────────────────────────────
    m.def("init_dmem", [](uint16_t node_id, int num_peers) {
        if (g_dmem) {
            throw std::runtime_error("DMEM already initialized");
        }
        DistributedMemoryManager::Config cfg;
        cfg.node_id   = node_id;
        cfg.num_peers = num_peers;
        g_dmem = std::make_unique<DistributedMemoryManager>(cfg);
        g_dmem->start();
    }, py::arg("node_id"), py::arg("num_peers"),
       "Initialize the distributed memory manager.");

    m.def("shutdown_dmem", []() {
        if (g_dmem) {
            g_dmem->shutdown();
            g_dmem.reset();
        }
    }, "Shut down the distributed memory manager.");

    // ── Region management ──────────────────────────────────────────────
    m.def("register_region",
          [](uint64_t addr, uint64_t length, uint8_t flags) -> uint32_t {
              return get_dmem()->register_region(
                  reinterpret_cast<void*>(addr), length, flags);
          },
          py::arg("addr"), py::arg("length"), py::arg("flags") = 0x03,
          "Register a memory region with the RDMA NIC. Returns region_id.");

    m.def("unregister_region",
          [](uint32_t region_id) {
              get_dmem()->unregister_region(region_id);
          },
          py::arg("region_id"),
          "Unregister a memory region.");

    // ── One-sided RDMA operations ──────────────────────────────────────
    m.def("put",
          [](uint16_t target_node, uint16_t target_region, uint64_t offset,
             uint64_t src_addr, uint64_t length) -> uint64_t {
              GlobalAddress dst{};
              dst.node_id   = target_node;
              dst.region_id = target_region;
              dst.offset    = offset;
              return get_dmem()->put(target_node, dst,
                                     reinterpret_cast<const void*>(src_addr),
                                     length);
          },
          py::arg("target_node"), py::arg("target_region"), py::arg("offset"),
          py::arg("src_addr"), py::arg("length"),
          "RDMA write: put local data to remote memory. Returns wr_id.");

    m.def("get",
          [](uint16_t target_node, uint16_t target_region, uint64_t offset,
             uint64_t dst_addr, uint64_t length) -> uint64_t {
              GlobalAddress src{};
              src.node_id   = target_node;
              src.region_id = target_region;
              src.offset    = offset;
              return get_dmem()->get(target_node, src,
                                     reinterpret_cast<void*>(dst_addr),
                                     length);
          },
          py::arg("target_node"), py::arg("target_region"), py::arg("offset"),
          py::arg("dst_addr"), py::arg("length"),
          "RDMA read: pull remote data into local memory. Returns wr_id.");

    m.def("poll_completion",
          [](uint64_t wr_id, int timeout_ms) -> bool {
              return get_dmem()->poll_completion(
                  wr_id, std::chrono::milliseconds(timeout_ms));
          },
          py::arg("wr_id"), py::arg("timeout_ms") = 5000,
          "Block until the RDMA operation identified by wr_id completes.");

    m.def("drain_pending", []() {
        get_dmem()->drain_pending();
    }, "Wait for all pending RDMA operations to complete.");

    // ── Statistics ────────────────────────────────────────────────────
    m.def("get_stats", []() -> py::dict {
        auto* dmem = get_dmem();
        auto s = dmem->stats();
        py::dict d;
        d["total_put_bytes"] = s.total_put_bytes;
        d["total_get_bytes"] = s.total_get_bytes;
        d["total_put_ops"]   = s.total_put_ops;
        d["total_get_ops"]   = s.total_get_ops;
        d["total_errors"]    = s.total_errors;
        return d;
    }, "Get DMEM statistics.");

    // ── Metal kernel access (for Python-side reduction) ────────────────
    m.def("metal_kernels_init", &metal_kernels_init,
          "Initialize the Metal kernel cache (must be called before any GPU ops).");

    m.def("metal_sync", &metal_sync,
          "Block until all enqueued Metal commands complete.");
}

} // namespace distro
