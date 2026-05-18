"""Background sampler: every `interval` seconds, snapshot per-core CPU,
RSS, and torch.mps.driver_allocated_memory(). Returns the time series
as a dict of lists when joined."""
from __future__ import annotations

import threading
import time
from typing import List, Tuple

import psutil
import torch


class Monitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.t: List[float] = []
        self.cpu_total: List[float] = []          # 0..100*ncpu
        self.cpu_per_core: List[List[float]] = []
        self.mps_driver_mb: List[float] = []
        self.mps_alloc_mb: List[float] = []
        self.proc = psutil.Process()
        self.proc_cpu_pct: List[float] = []

    def _loop(self, t0):
        # Prime per-core to avoid the first-call-returns-0 issue
        psutil.cpu_percent(interval=None, percpu=True)
        self.proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.t.append(time.perf_counter() - t0)
            self.cpu_per_core.append(
                psutil.cpu_percent(interval=None, percpu=True))
            self.cpu_total.append(sum(self.cpu_per_core[-1]))
            self.proc_cpu_pct.append(self.proc.cpu_percent(interval=None))
            try:
                self.mps_driver_mb.append(
                    torch.mps.driver_allocated_memory() / 1e6)
                self.mps_alloc_mb.append(
                    torch.mps.current_allocated_memory() / 1e6)
            except Exception:
                self.mps_driver_mb.append(0.0)
                self.mps_alloc_mb.append(0.0)
            self._stop.wait(self.interval)

    def start(self):
        t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, args=(t0,), daemon=True, name="monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self.t:
            return {}
        ncpu = len(self.cpu_per_core[0]) if self.cpu_per_core else 1
        # Avg/peak across the sampled window.
        return {
            "n_samples": len(self.t),
            "ncpu": ncpu,
            "cpu_total_avg_pct": sum(self.cpu_total) / len(self.cpu_total) / ncpu,
            "cpu_total_peak_pct": max(self.cpu_total) / ncpu,
            "proc_cpu_avg_pct": sum(self.proc_cpu_pct) / len(self.proc_cpu_pct),
            "proc_cpu_peak_pct": max(self.proc_cpu_pct),
            "mps_driver_peak_mb": max(self.mps_driver_mb),
            "mps_alloc_peak_mb": max(self.mps_alloc_mb),
            "duration_s": self.t[-1],
        }
