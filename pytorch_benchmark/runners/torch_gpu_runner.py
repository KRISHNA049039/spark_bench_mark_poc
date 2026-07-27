"""
Torch + GPU Runner

Executes training and inference on CUDA GPU using PyTorch.
Tracks GPU-specific resource utilization: VRAM, GPU utilization, memory
allocation/deallocation, CUDA events for timing, and GC.
"""

import gc
import time
from typing import Dict, Any

import torch
import psutil

from pytorch_benchmark.config import RANDOM_SEED, EPOCHS, LEARNING_RATE, BATCH_SIZE
from pytorch_benchmark.data_generation import seed_everything, get_structured_datasets, get_unstructured_datasets, make_dataloader
from pytorch_benchmark.models import create_model, get_model_summary
from pytorch_benchmark.runners.base_runner import BaseRunner, RunnerResult


class TorchGPURunner(BaseRunner):
    """
    Runner for PyTorch on CUDA GPU.

    Optimizations applied:
    - pin_memory for faster host-to-device transfers
    - non_blocking data transfers
    - torch.cuda.amp mixed precision (optional)
    - cudnn.benchmark disabled for reproducibility
    - Proper CUDA memory tracking and cleanup
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        gpu_id: int = 0,
        use_amp: bool = False,
    ):
        self.gpu_id = gpu_id
        self.use_amp = use_amp
        super().__init__(mode="torch_gpu", seed=seed)

    def _get_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Cannot run torch_gpu mode. "
                "Install CUDA-enabled PyTorch or use torch_cpu mode."
            )
        return torch.device(f"cuda:{self.gpu_id}")

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to GPU with non_blocking for pinned memory."""
        return tensor.to(self.device, non_blocking=True)

    def run_full_benchmark(
        self,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark for both structured and unstructured data on GPU.

        Returns:
            Dictionary with results for each data type plus system info.
        """
        seed_everything(self.seed)

        # Warm up GPU
        self._gpu_warmup()

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
        """Run training + evaluation for a single data type on GPU."""
        seed_everything(self.seed)

        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        # --- Data preparation (pin_memory=True for efficient GPU transfer) ---
        if data_type == "structured":
            train_ds, test_ds, metadata = get_structured_datasets()
        else:
            train_ds, test_ds, metadata = get_unstructured_datasets()

        train_loader = make_dataloader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        test_loader = make_dataloader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # --- Model creation ---
        model = create_model(data_type, device=self.device, seed=self.seed)
        model_info = get_model_summary(model)

        # --- Resource tracking setup ---
        gc_stats_before = self._get_gc_stats()
        io_before = self._get_io_stats()
        mem_before_cpu = self._get_cpu_memory_stats()
        gpu_mem_before = self._get_gpu_memory_stats()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()

        # CUDA timing events for precise GPU measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # --- Run training + evaluation ---
        start_event.record()
        result = self.run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            data_type=data_type,
            epochs=epochs,
            lr=lr,
        )
        end_event.record()
        torch.cuda.synchronize(self.device)
        gpu_elapsed_ms = start_event.elapsed_time(end_event)

        # --- Collect post-run resource stats ---
        cpu_times_after = process.cpu_times()
        mem_after_cpu = self._get_cpu_memory_stats()
        gpu_mem_after = self._get_gpu_memory_stats()
        io_after = self._get_io_stats()
        gc_stats_after = self._get_gc_stats()

        # Build comprehensive resource metrics
        result.resource_metrics = {
            "model_info": model_info,
            "gpu": {
                "device_name": torch.cuda.get_device_name(self.device),
                "device_id": self.gpu_id,
                "gpu_elapsed_ms": gpu_elapsed_ms,
                "memory_before": gpu_mem_before,
                "memory_after": gpu_mem_after,
                "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024**2),
                "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024**2),
                "memory_allocated_delta_mb": (
                    gpu_mem_after["allocated_mb"] - gpu_mem_before["allocated_mb"]
                ),
                "memory_reserved_delta_mb": (
                    gpu_mem_after["reserved_mb"] - gpu_mem_before["reserved_mb"]
                ),
                "use_amp": self.use_amp,
            },
            "cpu": {
                "user_time_sec": cpu_times_after.user - cpu_times_before.user,
                "system_time_sec": cpu_times_after.system - cpu_times_before.system,
            },
            "memory_cpu": {
                "before": mem_before_cpu,
                "after": mem_after_cpu,
                "delta_rss_mb": mem_after_cpu["rss_mb"] - mem_before_cpu["rss_mb"],
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
                "gpu_total_time_ms": gpu_elapsed_ms,
            },
        }

        return result

    # ------------------------------------------------------------------
    # GPU-specific helpers
    # ------------------------------------------------------------------

    def _gpu_warmup(self):
        """
        Warm up the GPU to avoid cold-start latency affecting benchmarks.
        Runs a small tensor operation to initialize CUDA context.
        """
        warmup_tensor = torch.randn(256, 256, device=self.device)
        _ = torch.mm(warmup_tensor, warmup_tensor)
        torch.cuda.synchronize(self.device)
        del warmup_tensor
        torch.cuda.empty_cache()

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2),
            "free_mb": (
                torch.cuda.memory_reserved(self.device) -
                torch.cuda.memory_allocated(self.device)
            ) / (1024**2),
        }

    def _get_gpu_utilization(self) -> Dict[str, Any]:
        """Get GPU utilization using GPUtil if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus and self.gpu_id < len(gpus):
                gpu = gpus[self.gpu_id]
                return {
                    "gpu_util_percent": gpu.load * 100,
                    "memory_util_percent": gpu.memoryUtil * 100,
                    "temperature_c": gpu.temperature,
                }
        except ImportError:
            pass
        return {"gpu_util_percent": -1, "memory_util_percent": -1, "temperature_c": -1}

    # ------------------------------------------------------------------
    # Shared resource collection helpers
    # ------------------------------------------------------------------

    def _get_cpu_memory_stats(self) -> Dict[str, float]:
        """Get current process CPU memory usage."""
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "rss_mb": mem.rss / (1024**2),
            "vms_mb": mem.vms / (1024**2),
            "percent": process.memory_percent(),
        }

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collector statistics."""
        gc.collect()
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
            return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system + GPU information for the benchmark report."""
        gpu_info = {}
        try:
            gpu_info = {
                "name": torch.cuda.get_device_name(self.device),
                "capability": torch.cuda.get_device_capability(self.device),
                "total_memory_mb": torch.cuda.get_device_properties(self.device).total_memory / (1024**2),
            }
        except Exception:
            gpu_info = {"name": "unknown", "capability": (0, 0), "total_memory_mb": 0}

        return {
            "platform": {
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            },
            "torch": {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "gpu": gpu_info,
            "device": str(self.device),
            "mode": self.mode,
        }

    def _cleanup(self):
        """Force cleanup of GPU memory and garbage collection."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
