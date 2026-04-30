__version__ = "0.4.0"

COMPATIBILITY_MATRIX = {
    "macos": ["15.0", "15.1", "15.2", "15.3", "15.4", "26.2+"],
    "python": ["3.11", "3.12", "3.13"],
    "pytorch": ["2.5.x", "2.6.x"],
    "hardware": "Apple Silicon (M1/M2/M3/M4 family)",
    "interconnect": "Thunderbolt 5 (RDMA) or Thunderbolt 4/3 (IP)",
    "features": [
        "distributed_memory_manager",
        "one_sided_rdma",
        "metal_kernel_dispatch",
        "data_parallel_sharding",
        "tensor_parallel_sharding",
        "pipeline_parallel_sharding",
    ],
}
