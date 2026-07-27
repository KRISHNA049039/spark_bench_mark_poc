"""
Resource Monitoring and Metrics Collection Module

Provides continuous background monitoring of system resources during benchmark
execution. Collects time-series data for:
    - CPU utilization (per-core and aggregate)
    - Memory allocation and release (RSS, VMS, heap)
    - GPU utilization and VRAM (if available)
    - IO statistics (read/write bytes and ops)
    - Garbage collection events and overhead
    - Process-level metrics (threads, file descriptors)

Supports both instantaneous snapshots and time-series collection
with configurable sampling intervals.
"""

import gc
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import psutil
import torch

from pytorch_benchmark.config import MONITOR_INTERVAL_SEC


# ---------------------------------------------------------------------------
# Data classes for structured metrics
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """Single memory measurement."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_system_mb: float


@dataclass
class GPUSnapshot:
    """Single GPU measurement."""
    timestamp: float
    device_id: int
    allocated_mb: float
    reserved_mb: float
    utilization_percent: float = -1.0
    temperature_c: float = -1.0


@dataclass
class IOSnapshot:
    """Single IO measurement."""
    timestamp: float
    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int


@dataclass
class CPUSnapshot:
    """Single CPU measurement."""
    timestamp: float
    percent_per_core: List[float]
    percent_total: float
    num_threads: int
    context_switches: int = 0


@dataclass
class GCSnapshot:
    """GC state at a point in time."""
    timestamp: float
    gen0_collections: int
    gen1_collections: int
    gen2_collections: int
    total_objects: int
    garbage_objects: int


# ---------------------------------------------------------------------------
# Resource Monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    Background thread that periodically samples system resource utilization.

    Usage:
        monitor = ResourceMonitor(interval=0.5)
        monitor.start()
        ... run workload ...
        monitor.stop()
        summary = monitor.get_summary()
        timeseries = monitor.get_timeseries()
    """

    def __init__(
        self,
        interval: float = MONITOR_INTERVAL_SEC,
        track_gpu: bool = True,
        track_per_core_cpu: bool = True,
    ):
        self.interval = interval
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_per_core_cpu = track_per_core_cpu

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()

        # Time-series storage
        self._memory_samples: List[MemorySnapshot] = []
        self._gpu_samples: List[GPUSnapshot] = []
        self._io_samples: List[IOSnapshot] = []
        self._cpu_samples: List[CPUSnapshot] = []
        self._gc_samples: List[GCSnapshot] = []

        # Start/stop timestamps
        self._start_time: float = 0.0
        self._stop_time: float = 0.0

        # Initial baselines
        self._baseline_io: Optional[Dict] = None
        self._baseline_gc: Optional[Dict] = None

    def start(self):
        """Start background resource monitoring."""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()

        # Record baselines
        self._baseline_io = self._sample_io_raw()
        self._baseline_gc = self._sample_gc_raw()

        # Start monitoring thread
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ResourceMonitor",
        )
        self._thread.start()

    def stop(self):
        """Stop background monitoring and join thread."""
        if not self._running:
            return

        self._running = False
        self._stop_time = time.perf_counter()

        if self._thread:
            self._thread.join(timeout=self.interval * 3)
            self._thread = None

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self._running:
            timestamp = time.perf_counter() - self._start_time

            try:
                self._sample_memory(timestamp)
                self._sample_cpu(timestamp)
                self._sample_io(timestamp)
                self._sample_gc(timestamp)

                if self.track_gpu:
                    self._sample_gpu(timestamp)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process may have ended or lost permissions
                break
            except Exception:
                # Don't crash monitoring for unexpected errors
                pass

            time.sleep(self.interval)

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def _sample_memory(self, timestamp: float):
        """Sample process and system memory."""
        mem_info = self._process.memory_info()
        sys_mem = psutil.virtual_memory()

        self._memory_samples.append(MemorySnapshot(
            timestamp=timestamp,
            rss_mb=mem_info.rss / (1024**2),
            vms_mb=mem_info.vms / (1024**2),
            percent=self._process.memory_percent(),
            available_system_mb=sys_mem.available / (1024**2),
        ))

    def _sample_cpu(self, timestamp: float):
        """Sample CPU utilization."""
        if self.track_per_core_cpu:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
        else:
            per_core = []

        total_percent = self._process.cpu_percent(interval=None)
        num_threads = self._process.num_threads()

        ctx_switches = 0
        try:
            ctx = self._process.num_ctx_switches()
            ctx_switches = ctx.voluntary + ctx.involuntary
        except (psutil.AccessDenied, AttributeError):
            pass

        self._cpu_samples.append(CPUSnapshot(
            timestamp=timestamp,
            percent_per_core=per_core,
            percent_total=total_percent,
            num_threads=num_threads,
            context_switches=ctx_switches,
        ))

    def _sample_io(self, timestamp: float):
        """Sample IO counters."""
        io_raw = self._sample_io_raw()
        self._io_samples.append(IOSnapshot(
            timestamp=timestamp,
            read_bytes=io_raw["read_bytes"],
            write_bytes=io_raw["write_bytes"],
            read_count=io_raw["read_count"],
            write_count=io_raw["write_count"],
        ))

    def _sample_io_raw(self) -> Dict[str, int]:
        """Get raw IO counters."""
        try:
            io = self._process.io_counters()
            return {
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "read_count": io.read_count,
                "write_count": io.write_count,
            }
        except (psutil.AccessDenied, AttributeError):
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}

    def _sample_gc(self, timestamp: float):
        """Sample garbage collector state (non-invasive, no forced collection)."""
        stats = gc.get_stats()
        self._gc_samples.append(GCSnapshot(
            timestamp=timestamp,
            gen0_collections=stats[0]["collections"],
            gen1_collections=stats[1]["collections"],
            gen2_collections=stats[2]["collections"],
            total_objects=len(gc.get_objects()) if len(self._gc_samples) % 10 == 0 else -1,
            garbage_objects=len(gc.garbage),
        ))

    def _sample_gc_raw(self) -> Dict[str, Any]:
        """Get raw GC stats."""
        stats = gc.get_stats()
        return {
            "collections": [s["collections"] for s in stats],
            "collected": [s["collected"] for s in stats],
            "uncollectable": [s["uncollectable"] for s in stats],
        }

    def _sample_gpu(self, timestamp: float):
        """Sample GPU memory and utilization."""
        for device_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_id) / (1024**2)

            utilization = -1.0
            temperature = -1.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if device_id < len(gpus):
                    utilization = gpus[device_id].load * 100
                    temperature = gpus[device_id].temperature
            except ImportError:
                pass

            self._gpu_samples.append(GPUSnapshot(
                timestamp=timestamp,
                device_id=device_id,
                allocated_mb=allocated,
                reserved_mb=reserved,
                utilization_percent=utilization,
                temperature_c=temperature,
            ))

    # ------------------------------------------------------------------
    # Results retrieval
    # ------------------------------------------------------------------

    def get_timeseries(self) -> Dict[str, Any]:
        """
        Return full time-series data for all monitored resources.

        Returns dict with keys: memory, cpu, gpu, io, gc
        Each contains a list of measurement dicts.
        """
        return {
            "memory": [
                {
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "percent": s.percent,
                    "available_system_mb": s.available_system_mb,
                }
                for s in self._memory_samples
            ],
            "cpu": [
                {
                    "timestamp": s.timestamp,
                    "percent_total": s.percent_total,
                    "num_threads": s.num_threads,
                    "context_switches": s.context_switches,
                    "percent_per_core": s.percent_per_core,
                }
                for s in self._cpu_samples
            ],
            "gpu": [
                {
                    "timestamp": s.timestamp,
                    "device_id": s.device_id,
                    "allocated_mb": s.allocated_mb,
                    "reserved_mb": s.reserved_mb,
                    "utilization_percent": s.utilization_percent,
                    "temperature_c": s.temperature_c,
                }
                for s in self._gpu_samples
            ],
            "io": [
                {
                    "timestamp": s.timestamp,
                    "read_bytes": s.read_bytes,
                    "write_bytes": s.write_bytes,
                    "read_count": s.read_count,
                    "write_count": s.write_count,
                }
                for s in self._io_samples
            ],
            "gc": [
                {
                    "timestamp": s.timestamp,
                    "gen0_collections": s.gen0_collections,
                    "gen1_collections": s.gen1_collections,
                    "gen2_collections": s.gen2_collections,
                    "total_objects": s.total_objects,
                    "garbage_objects": s.garbage_objects,
                }
                for s in self._gc_samples
            ],
            "duration_sec": self._stop_time - self._start_time if self._stop_time else 0,
            "num_samples": len(self._memory_samples),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Return aggregated summary statistics for the monitoring period.

        Computes min/max/mean/final for each metric category.
        """
        duration = self._stop_time - self._start_time if self._stop_time else 0
        summary = {
            "duration_sec": duration,
            "num_samples": len(self._memory_samples),
            "sampling_interval_sec": self.interval,
        }

        # --- Memory summary ---
        if self._memory_samples:
            rss_values = [s.rss_mb for s in self._memory_samples]
            vms_values = [s.vms_mb for s in self._memory_samples]
            summary["memory"] = {
                "rss_mb_min": min(rss_values),
                "rss_mb_max": max(rss_values),
                "rss_mb_mean": np.mean(rss_values),
                "rss_mb_final": rss_values[-1],
                "rss_mb_delta": rss_values[-1] - rss_values[0],
                "vms_mb_max": max(vms_values),
                "memory_released": rss_values[-1] < max(rss_values),
            }

        # --- CPU summary ---
        if self._cpu_samples:
            cpu_values = [s.percent_total for s in self._cpu_samples]
            thread_values = [s.num_threads for s in self._cpu_samples]
            ctx_values = [s.context_switches for s in self._cpu_samples]
            summary["cpu"] = {
                "percent_min": min(cpu_values),
                "percent_max": max(cpu_values),
                "percent_mean": np.mean(cpu_values),
                "threads_max": max(thread_values),
                "threads_mean": np.mean(thread_values),
                "context_switches_total": max(ctx_values) - min(ctx_values) if ctx_values else 0,
            }

            # Per-core utilization efficiency
            if self.track_per_core_cpu and self._cpu_samples[0].percent_per_core:
                all_cores = [s.percent_per_core for s in self._cpu_samples if s.percent_per_core]
                if all_cores:
                    core_means = np.mean(all_cores, axis=0)
                    summary["cpu"]["per_core_mean_percent"] = core_means.tolist()
                    summary["cpu"]["core_utilization_balance"] = float(
                        np.std(core_means) / (np.mean(core_means) + 1e-8)
                    )

        # --- GPU summary ---
        if self._gpu_samples:
            # Group by device
            devices = set(s.device_id for s in self._gpu_samples)
            gpu_summary = {}
            for dev_id in devices:
                dev_samples = [s for s in self._gpu_samples if s.device_id == dev_id]
                alloc_values = [s.allocated_mb for s in dev_samples]
                reserved_values = [s.reserved_mb for s in dev_samples]
                util_values = [s.utilization_percent for s in dev_samples if s.utilization_percent >= 0]

                gpu_summary[f"gpu_{dev_id}"] = {
                    "allocated_mb_min": min(alloc_values),
                    "allocated_mb_max": max(alloc_values),
                    "allocated_mb_mean": np.mean(alloc_values),
                    "allocated_mb_final": alloc_values[-1],
                    "reserved_mb_max": max(reserved_values),
                    "utilization_percent_mean": np.mean(util_values) if util_values else -1,
                    "utilization_percent_max": max(util_values) if util_values else -1,
                    "memory_released": alloc_values[-1] < max(alloc_values),
                }
            summary["gpu"] = gpu_summary

        # --- IO summary ---
        if self._io_samples and self._baseline_io:
            final_io = self._sample_io_raw()
            summary["io"] = {
                "total_read_bytes": final_io["read_bytes"] - self._baseline_io["read_bytes"],
                "total_write_bytes": final_io["write_bytes"] - self._baseline_io["write_bytes"],
                "total_read_ops": final_io["read_count"] - self._baseline_io["read_count"],
                "total_write_ops": final_io["write_count"] - self._baseline_io["write_count"],
                "read_throughput_mb_s": (
                    (final_io["read_bytes"] - self._baseline_io["read_bytes"])
                    / (1024**2)
                    / max(duration, 0.001)
                ),
                "write_throughput_mb_s": (
                    (final_io["write_bytes"] - self._baseline_io["write_bytes"])
                    / (1024**2)
                    / max(duration, 0.001)
                ),
            }

        # --- GC summary ---
        if self._gc_samples and self._baseline_gc:
            final_gc = self._sample_gc_raw()
            summary["gc"] = {
                "gen0_collections": final_gc["collections"][0] - self._baseline_gc["collections"][0],
                "gen1_collections": final_gc["collections"][1] - self._baseline_gc["collections"][1],
                "gen2_collections": final_gc["collections"][2] - self._baseline_gc["collections"][2],
                "total_collected": (
                    sum(final_gc["collected"]) - sum(self._baseline_gc["collected"])
                ),
                "uncollectable": sum(final_gc["uncollectable"]),
                "garbage_objects_final": len(gc.garbage),
            }

        return summary

    def reset(self):
        """Reset all collected data for reuse."""
        self._memory_samples.clear()
        self._gpu_samples.clear()
        self._io_samples.clear()
        self._cpu_samples.clear()
        self._gc_samples.clear()
        self._baseline_io = None
        self._baseline_gc = None
        self._start_time = 0.0
        self._stop_time = 0.0


# ---------------------------------------------------------------------------
# Standalone utility functions for one-shot metrics
# ---------------------------------------------------------------------------

def get_system_snapshot() -> Dict[str, Any]:
    """
    Take a single snapshot of all system resources.
    Useful for before/after comparisons without continuous monitoring.
    """
    process = psutil.Process()
    mem = process.memory_info()
    sys_mem = psutil.virtual_memory()

    snapshot = {
        "timestamp": time.time(),
        "process": {
            "pid": process.pid,
            "rss_mb": mem.rss / (1024**2),
            "vms_mb": mem.vms / (1024**2),
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads(),
            "num_fds": _get_num_fds(process),
        },
        "system": {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_mb": sys_mem.total / (1024**2),
            "memory_available_mb": sys_mem.available / (1024**2),
            "memory_used_percent": sys_mem.percent,
        },
        "gc": {
            "gen0_collections": gc.get_stats()[0]["collections"],
            "gen1_collections": gc.get_stats()[1]["collections"],
            "gen2_collections": gc.get_stats()[2]["collections"],
            "tracked_objects": len(gc.get_objects()),
            "garbage_objects": len(gc.garbage),
            "thresholds": gc.get_threshold(),
        },
    }

    # IO
    try:
        io = process.io_counters()
        snapshot["io"] = {
            "read_bytes": io.read_bytes,
            "write_bytes": io.write_bytes,
            "read_count": io.read_count,
            "write_count": io.write_count,
        }
    except (psutil.AccessDenied, AttributeError):
        snapshot["io"] = {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}

    # GPU
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_mb": torch.cuda.memory_allocated(i) / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved(i) / (1024**2),
                "total_mb": torch.cuda.get_device_properties(i).total_memory / (1024**2),
            })
        snapshot["gpu"] = gpu_info

    return snapshot


def compare_snapshots(before: Dict, after: Dict) -> Dict[str, Any]:
    """
    Compare two system snapshots and return the deltas.
    """
    delta = {
        "duration_sec": after["timestamp"] - before["timestamp"],
        "memory": {
            "rss_delta_mb": after["process"]["rss_mb"] - before["process"]["rss_mb"],
            "vms_delta_mb": after["process"]["vms_mb"] - before["process"]["vms_mb"],
            "system_available_delta_mb": (
                after["system"]["memory_available_mb"] -
                before["system"]["memory_available_mb"]
            ),
        },
        "gc": {
            "gen0_collections": (
                after["gc"]["gen0_collections"] - before["gc"]["gen0_collections"]
            ),
            "gen1_collections": (
                after["gc"]["gen1_collections"] - before["gc"]["gen1_collections"]
            ),
            "gen2_collections": (
                after["gc"]["gen2_collections"] - before["gc"]["gen2_collections"]
            ),
            "tracked_objects_delta": (
                after["gc"]["tracked_objects"] - before["gc"]["tracked_objects"]
            ),
        },
        "io": {
            "read_bytes": after["io"]["read_bytes"] - before["io"]["read_bytes"],
            "write_bytes": after["io"]["write_bytes"] - before["io"]["write_bytes"],
            "read_count": after["io"]["read_count"] - before["io"]["read_count"],
            "write_count": after["io"]["write_count"] - before["io"]["write_count"],
        },
        "threads_delta": (
            after["process"]["num_threads"] - before["process"]["num_threads"]
        ),
    }

    # GPU deltas
    if "gpu" in before and "gpu" in after:
        gpu_delta = []
        for i, (b_gpu, a_gpu) in enumerate(zip(before["gpu"], after["gpu"])):
            gpu_delta.append({
                "device_id": i,
                "allocated_delta_mb": a_gpu["allocated_mb"] - b_gpu["allocated_mb"],
                "reserved_delta_mb": a_gpu["reserved_mb"] - b_gpu["reserved_mb"],
            })
        delta["gpu"] = gpu_delta

    return delta


def _get_num_fds(process: psutil.Process) -> int:
    """Get number of file descriptors (cross-platform)."""
    try:
        return process.num_fds()
    except AttributeError:
        # Windows doesn't have num_fds, use num_handles instead
        try:
            return process.num_handles()
        except (AttributeError, psutil.AccessDenied):
            return -1


# ---------------------------------------------------------------------------
# Memory profiler for allocation tracking
# ---------------------------------------------------------------------------

class MemoryTracker:
    """
    Track memory allocations and releases during a code block.

    Usage:
        tracker = MemoryTracker()
        tracker.checkpoint("before_model")
        model = create_model(...)
        tracker.checkpoint("after_model")
        tracker.checkpoint("after_training")
        report = tracker.report()
    """

    def __init__(self):
        self._checkpoints: List[tuple] = []  # (name, snapshot)

    def checkpoint(self, name: str):
        """Record a named memory checkpoint."""
        process = psutil.Process()
        mem = process.memory_info()

        gpu_allocated = 0.0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)

        self._checkpoints.append((name, {
            "rss_mb": mem.rss / (1024**2),
            "vms_mb": mem.vms / (1024**2),
            "gpu_allocated_mb": gpu_allocated,
            "gc_objects": len(gc.get_objects()),
            "timestamp": time.perf_counter(),
        }))

    def report(self) -> Dict[str, Any]:
        """Generate a report of memory changes between checkpoints."""
        if len(self._checkpoints) < 2:
            return {"error": "Need at least 2 checkpoints"}

        transitions = []
        for i in range(1, len(self._checkpoints)):
            prev_name, prev_snap = self._checkpoints[i - 1]
            curr_name, curr_snap = self._checkpoints[i]
            transitions.append({
                "from": prev_name,
                "to": curr_name,
                "rss_delta_mb": curr_snap["rss_mb"] - prev_snap["rss_mb"],
                "vms_delta_mb": curr_snap["vms_mb"] - prev_snap["vms_mb"],
                "gpu_delta_mb": curr_snap["gpu_allocated_mb"] - prev_snap["gpu_allocated_mb"],
                "gc_objects_delta": curr_snap["gc_objects"] - prev_snap["gc_objects"],
                "time_elapsed_sec": curr_snap["timestamp"] - prev_snap["timestamp"],
            })

        first_snap = self._checkpoints[0][1]
        last_snap = self._checkpoints[-1][1]

        return {
            "checkpoints": [(name, snap) for name, snap in self._checkpoints],
            "transitions": transitions,
            "total_rss_delta_mb": last_snap["rss_mb"] - first_snap["rss_mb"],
            "total_gpu_delta_mb": last_snap["gpu_allocated_mb"] - first_snap["gpu_allocated_mb"],
            "total_time_sec": last_snap["timestamp"] - first_snap["timestamp"],
            "memory_released_at_end": last_snap["rss_mb"] < max(
                s["rss_mb"] for _, s in self._checkpoints
            ),
        }
