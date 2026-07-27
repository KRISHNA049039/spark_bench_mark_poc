"""
Torch + CPU Runner

Executes training and inference entirely on CPU using PyTorch.
Tracks CPU-specific resource utilization: memory, threads, IO, GC.
"""

import gc
import time
import threading
from typing import Dict, Any

import torch
import psutil

from pytorch_benchmark.config import RANDOM_SEED, EPOCHS, LEARNING_RATE, BATCH_SIZE
from pytorch_benchmark.data_generation import seed_everything, get_structured_datasets, get_unstructured_datasets, make_dataloader
from pytorch_benchmark.models import create_model, get_model_summary
from pytorch_benchmark.runners.base_runner import BaseRunner, RunnerResult


class TorchCPURunner(BaseRunner):
    """
    Runner for PyTorch on CPU.

    Optimizations applied:
    - torch.set_num_threads for efficient CPU utilization
    - Memory-efficient gradient computation
    - Proper cleanup and GC tracking
    """

    def __init__(self, seed: int = RANDOM_SEED, num_threads: int = None):
        self.num_threads = num_threads
        super().__init__(mode="torch_cpu", seed=seed)

        # Set CPU thread count for optimal utilization
        if num_threads is None:
            # Use physical cores (not hyperthreads) for best performance
            self.num_threads = psutil.cpu_count(logical=False) or 4
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(max(1, self.num_threads // 2))

    def _get_device(self) -> torch.device:
        return torch.device("cpu")

    def run_full_benchmark(
        self,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark for both structured and unstructured data.

        Returns:
            Dictionary with results for each data type plus system info.
        """
        seed_everything(self.seed)
        results = {}
        system_info = self._collect_system_info()

        for data_type in ("structured", "unstructured"):
            result = self._run_single(data_type, epochs, lr, batch_size)
            results[data_type] = result

        return {
            "mode": self.mode,
            "system_info": system_info,
            "structured": results["structured"].to_dict(),
            "unstructured": results["unstructured"].to_dict(),
        }

    def _run_single(
        self,
        data_type: str,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> RunnerResult:
        """Run training + evaluation for a single data type."""
        seed_everything(self.seed)

        # --- Data preparation ---
        if data_type == "structured":
            train_ds, test_ds, metadata = get_structured_datasets()
        else:
            train_ds, test_ds, metadata = get_unstructured_datasets()

        train_loader = make_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
        test_loader = make_dataloader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

        # --- Model creation ---
        model = create_model(data_type, device=self.device, seed=self.seed)
        model_info = get_model_summary(model)

        # --- Resource tracking setup ---
        gc_stats_before = self._get_gc_stats()
        io_before = self._get_io_stats()
        mem_before = self._get_memory_stats()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()

        # --- Run training + evaluation ---
        result = self.run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            data_type=data_type,
            epochs=epochs,
            lr=lr,
        )

        # --- Collect post-run resource stats ---
        cpu_times_after = process.cpu_times()
        mem_after = self._get_memory_stats()
        io_after = self._get_io_stats()
        gc_stats_after = self._get_gc_stats()

        # Build comprehensive resource metrics
        result.resource_metrics = {
            "model_info": model_info,
            "cpu": {
                "num_threads_configured": self.num_threads,
                "user_time_sec": cpu_times_after.user - cpu_times_before.user,
                "system_time_sec": cpu_times_after.system - cpu_times_before.system,
                "cpu_percent": process.cpu_percent(interval=0.1),
            },
            "memory": {
                "before": mem_before,
                "after": mem_after,
                "delta_rss_mb": mem_after["rss_mb"] - mem_before["rss_mb"],
                "delta_vms_mb": mem_after["vms_mb"] - mem_before["vms_mb"],
                "peak_rss_mb": mem_after["rss_mb"],  # approximation
            },
            "gc": {
                "before": gc_stats_before,
                "after": gc_stats_after,
                "collections_gen0": gc_stats_after["collections"][0] - gc_stats_before["collections"][0],
                "collections_gen1": gc_stats_after["collections"][1] - gc_stats_before["collections"][1],
                "collections_gen2": gc_stats_after["collections"][2] - gc_stats_before["collections"][2],
                "objects_collected": gc_stats_after["total_collected"] - gc_stats_before["total_collected"],
            },
            "io": {
                "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
                "read_count": io_after["read_count"] - io_before["read_count"],
                "write_count": io_after["write_count"] - io_before["write_count"],
            },
            "timing": {
                "total_train_time": result.total_train_time,
                "total_inference_time": result.total_inference_time,
                "avg_epoch_time": sum(result.epoch_times) / len(result.epoch_times) if result.epoch_times else 0,
                "epoch_times": result.epoch_times,
            },
        }

        return result

    # ------------------------------------------------------------------
    # Resource collection helpers
    # ------------------------------------------------------------------

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current process memory usage."""
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collector statistics."""
        gc.collect()  # force collection to get accurate counts
        stats = gc.get_stats()
        return {
            "collections": [s["collections"] for s in stats],
            "total_collected": sum(s["collected"] for s in stats),
            "uncollectable": sum(s["uncollectable"] for s in stats),
            "gc_enabled": gc.isenabled(),
            "thresholds": gc.get_threshold(),
        }

    def _get_io_stats(self) -> Dict[str, int]:
        """Get IO counters for the process."""
        process = psutil.Process()
        try:
            io = process.io_counters()
            return {
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "read_count": io.read_count,
                "write_count": io.write_count,
            }
        except (psutil.AccessDenied, AttributeError):
            # Some platforms may not support io_counters
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for the benchmark report."""
        return {
            "platform": {
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            },
            "torch": {
                "version": torch.__version__,
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
            },
            "device": str(self.device),
            "mode": self.mode,
        }
