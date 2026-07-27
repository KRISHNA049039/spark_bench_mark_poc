"""
Spark + GPU Runner

Distributes PyTorch model training and inference across Spark executors
with GPU acceleration. Each executor uses a CUDA device for computation.

Uses Spark's resource scheduling for GPU allocation and tracks both
GPU VRAM and Spark executor metrics.
"""

import gc
import os
import time
import pickle
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import psutil

from pyspark.sql import SparkSession
from pyspark import SparkContext

from pytorch_benchmark.config import (
    RANDOM_SEED,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    SPARK_MASTER,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_EXECUTOR_CORES,
    SPARK_NUM_EXECUTORS,
    STRUCTURED_NUM_FEATURES,
    STRUCTURED_NUM_CLASSES,
    UNSTRUCTURED_IMAGE_SIZE,
    UNSTRUCTURED_NUM_CLASSES,
)
from pytorch_benchmark.data_generation import (
    seed_everything,
    generate_structured_data,
    generate_unstructured_data,
)
from pytorch_benchmark.models import create_model, get_model_summary, TabularNet, ImageCNN
from pytorch_benchmark.runners.base_runner import RunnerResult, _hash_array


class SparkGPURunner:
    """
    Runner for distributed PyTorch training/inference using PySpark with GPU.

    Architecture:
        - Driver: Coordinates training, aggregates gradients, holds master model
        - Workers/Executors: Each gets a GPU device, processes data partitions
          with CUDA acceleration

    Strategy:
        - Data-parallel with GPU-accelerated gradient computation on each executor
        - Model weights broadcast to executors, loaded onto local GPU
        - Gradients computed on GPU, transferred back to driver for aggregation
        - Inference distributed across GPU-equipped executors

    GPU Resource Management:
        - Spark resource discovery for GPU allocation
        - Per-executor CUDA device assignment via partition index
        - Memory tracking for both GPU VRAM and executor heap
        - Proper CUDA cleanup after each operation
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        spark_master: str = SPARK_MASTER,
        driver_memory: str = SPARK_DRIVER_MEMORY,
        executor_memory: str = SPARK_EXECUTOR_MEMORY,
        executor_cores: int = SPARK_EXECUTOR_CORES,
        num_executors: int = SPARK_NUM_EXECUTORS,
        gpus_per_executor: int = 1,
    ):
        self.mode = "spark_gpu"
        self.seed = seed
        self.spark_master = spark_master
        self.driver_memory = driver_memory
        self.executor_memory = executor_memory
        self.executor_cores = executor_cores
        self.num_executors = num_executors
        self.gpus_per_executor = gpus_per_executor
        self.device = torch.device("cpu")  # Driver stays on CPU
        self.spark: SparkSession = None

    def _create_spark_session(self) -> SparkSession:
        """
        Create a SparkSession configured for GPU executors.

        Configures Spark resource scheduling to allocate GPUs to executors.
        Falls back gracefully if GPU scheduling is not available (e.g., local mode).
        """
        builder = (
            SparkSession.builder
            .master(self.spark_master)
            .appName("PyTorchBenchmark_SparkGPU")
            .config("spark.driver.memory", self.driver_memory)
            .config("spark.executor.memory", self.executor_memory)
            .config("spark.executor.cores", str(self.executor_cores))
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.python.worker.reuse", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Memory management
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            # GC configuration
            .config("spark.executor.extraJavaOptions",
                    "-XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
            .config("spark.driver.extraJavaOptions",
                    "-XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
            # Shuffle optimization
            .config("spark.shuffle.compress", "true")
            .config("spark.shuffle.spill.compress", "true")
            # GPU resource scheduling (Spark 3.x+)
            .config("spark.executor.resource.gpu.amount", str(self.gpus_per_executor))
            .config("spark.task.resource.gpu.amount", str(self.gpus_per_executor))
            # Larger off-heap for GPU data transfers
            .config("spark.executor.memoryOverhead", "1g")
        )

        if self.spark_master != "local[*]" and not self.spark_master.startswith("local"):
            builder = builder.config("spark.executor.instances", str(self.num_executors))

        return builder.getOrCreate()

    def _stop_spark(self):
        """Stop SparkSession and cleanup."""
        if self.spark:
            self.spark.stop()
            self.spark = None

    def run_full_benchmark(
        self,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark for both structured and unstructured data on Spark+GPU.

        Returns:
            Dictionary with results for each data type plus system/Spark/GPU info.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Cannot run spark_gpu mode. "
                "Ensure GPU-enabled PyTorch is installed and GPUs are accessible."
            )

        seed_everything(self.seed)
        self.spark = self._create_spark_session()

        try:
            system_info = self._collect_system_info()
            results = {}

            for data_type in ("structured", "unstructured"):
                result = self._run_single(data_type, epochs, lr, batch_size)
                results[data_type] = result

            return {
                "mode": self.mode,
                "system_info": system_info,
                "structured": results["structured"].to_dict(),
                "unstructured": results["unstructured"].to_dict(),
            }
        finally:
            self._stop_spark()

    def _run_single(
        self,
        data_type: str,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> RunnerResult:
        """Run GPU-accelerated distributed training + evaluation."""
        seed_everything(self.seed)
        result = RunnerResult(self.mode, data_type)

        # --- Data generation on driver ---
        if data_type == "structured":
            X, y = generate_structured_data()
        else:
            X, y = generate_unstructured_data()

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- Model creation (same seed as other runners for reproducibility) ---
        model = create_model(data_type, device=self.device, seed=self.seed)
        model_info = get_model_summary(model)

        # --- Resource tracking ---
        gc_stats_before = self._get_gc_stats()
        io_before = self._get_io_stats()
        mem_before = self._get_memory_stats()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()

        # Driver-side GPU memory tracking (if driver has GPU access)
        gpu_mem_before = self._get_gpu_memory_stats()

        # --- Distributed Training with GPU ---
        train_start = time.perf_counter()

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            loss, acc = self._distributed_train_epoch_gpu(
                model, X_train, y_train, data_type, batch_size, lr
            )
            epoch_time = time.perf_counter() - epoch_start

            result.train_losses.append(loss)
            result.train_accuracies.append(acc)
            result.epoch_times.append(epoch_time)

        result.total_train_time = time.perf_counter() - train_start

        # --- Distributed Inference with GPU ---
        infer_start = time.perf_counter()
        test_loss, test_acc, preds, probs = self._distributed_evaluate_gpu(
            model, X_test, y_test, data_type, batch_size
        )
        result.total_inference_time = time.perf_counter() - infer_start

        result.test_loss = test_loss
        result.test_accuracy = test_acc
        result.predictions = preds
        result.probabilities = probs

        # Model state hash
        result.model_state_hash = self._compute_model_hash(model)

        # --- Collect post-run resource stats ---
        cpu_times_after = process.cpu_times()
        mem_after = self._get_memory_stats()
        io_after = self._get_io_stats()
        gc_stats_after = self._get_gc_stats()
        gpu_mem_after = self._get_gpu_memory_stats()

        # Spark-specific metrics
        spark_metrics = self._collect_spark_metrics()

        result.resource_metrics = {
            "model_info": model_info,
            "driver": {
                "cpu_user_time_sec": cpu_times_after.user - cpu_times_before.user,
                "cpu_system_time_sec": cpu_times_after.system - cpu_times_before.system,
                "memory_before": mem_before,
                "memory_after": mem_after,
                "memory_delta_rss_mb": mem_after["rss_mb"] - mem_before["rss_mb"],
            },
            "executor": {
                "num_executors": self.num_executors,
                "executor_cores": self.executor_cores,
                "executor_memory": self.executor_memory,
                "gpus_per_executor": self.gpus_per_executor,
            },
            "gpu": {
                "driver_gpu_memory_before": gpu_mem_before,
                "driver_gpu_memory_after": gpu_mem_after,
                "num_gpus_available": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
            },
            "spark": spark_metrics,
            "gc": {
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
    # GPU-accelerated distributed training
    # ------------------------------------------------------------------

    def _distributed_train_epoch_gpu(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        data_type: str,
        batch_size: int,
        lr: float,
    ) -> tuple:
        """
        Distributed GPU training epoch using Spark.

        Each executor:
        1. Receives model weights via broadcast
        2. Loads model onto its assigned CUDA device
        3. Computes forward/backward on its data partition using GPU
        4. Returns gradients (moved to CPU for network transfer)

        Driver aggregates gradients and updates master model.
        """
        sc = self.spark.sparkContext
        num_partitions = max(2, sc.defaultParallelism)

        # Serialize model state for broadcast
        model_state_bytes = pickle.dumps(model.state_dict())
        model_state_bc = sc.broadcast(model_state_bytes)

        # Partition data with deterministic shuffling
        n_samples = len(X)
        indices = list(range(n_samples))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        chunk_size = (n_samples + num_partitions - 1) // num_partitions
        data_chunks = []
        for i in range(num_partitions):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk_indices = indices[start:end]
            data_chunks.append((i, X[chunk_indices], y[chunk_indices]))

        data_rdd = sc.parallelize(data_chunks, numSlices=num_partitions)

        config = {
            "data_type": data_type,
            "batch_size": batch_size,
            "lr": lr,
            "seed": self.seed,
            "n_features": STRUCTURED_NUM_FEATURES if data_type == "structured" else None,
            "n_classes": STRUCTURED_NUM_CLASSES if data_type == "structured" else UNSTRUCTURED_NUM_CLASSES,
            "image_size": None if data_type == "structured" else UNSTRUCTURED_IMAGE_SIZE,
        }
        config_bc = sc.broadcast(config)

        def compute_gradients_on_gpu(data_chunk):
            """
            Worker function: compute gradients on assigned GPU.

            GPU assignment strategy:
            - Uses partition index modulo available GPUs
            - Falls back to CPU if CUDA not available on worker
            """
            import torch
            import torch.nn as nn
            import numpy as np

            partition_id, X_local, y_local = data_chunk
            cfg = config_bc.value
            state_bytes = model_state_bc.value

            # Determine GPU device for this executor
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                device_id = partition_id % gpu_count
                device = torch.device(f"cuda:{device_id}")

                # Set deterministic behavior
                torch.cuda.manual_seed_all(cfg["seed"])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                device = torch.device("cpu")

            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])

            # Recreate model on worker's GPU
            if cfg["data_type"] == "structured":
                local_model = TabularNet(
                    n_features=cfg["n_features"],
                    n_classes=cfg["n_classes"],
                )
            else:
                local_model = ImageCNN(
                    image_size=cfg["image_size"],
                    n_classes=cfg["n_classes"],
                )

            local_model.load_state_dict(pickle.loads(state_bytes))
            local_model = local_model.to(device)
            local_model.train()

            # Create tensors and move to GPU
            X_tensor = torch.from_numpy(X_local).to(device)
            y_tensor = torch.from_numpy(y_local).to(device)

            # Forward + backward on GPU
            criterion = nn.CrossEntropyLoss()
            bs = cfg["batch_size"]
            n = len(X_local)

            total_loss = 0.0
            correct = 0
            total = 0

            local_model.zero_grad()

            for start_idx in range(0, n, bs):
                end_idx = min(start_idx + bs, n)
                bx = X_tensor[start_idx:end_idx]
                by = y_tensor[start_idx:end_idx]

                outputs = local_model(bx)
                loss = criterion(outputs, by)
                loss.backward()

                total_loss += loss.item() * (end_idx - start_idx)
                _, predicted = outputs.max(1)
                total += (end_idx - start_idx)
                correct += predicted.eq(by).sum().item()

            # Move gradients back to CPU for network transfer
            grads = {}
            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    grads[name] = param.grad.cpu().numpy().copy()

            # GPU memory stats from this executor
            gpu_stats = {}
            if device.type == "cuda":
                gpu_stats = {
                    "device_id": device_id,
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
                }
                # Cleanup GPU memory on executor
                del local_model, X_tensor, y_tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

            return (grads, total_loss, correct, total, gpu_stats)

        # Execute on Spark with GPU
        partition_results = data_rdd.map(compute_gradients_on_gpu).collect()

        # Aggregate gradients on driver
        total_loss = sum(r[1] for r in partition_results)
        total_correct = sum(r[2] for r in partition_results)
        total_samples = sum(r[3] for r in partition_results)

        # Weighted average gradients
        avg_grads = {}
        for result_tuple in partition_results:
            grads = result_tuple[0]
            weight = result_tuple[3] / total_samples
            for name, grad in grads.items():
                if name not in avg_grads:
                    avg_grads[name] = np.zeros_like(grad)
                avg_grads[name] += grad * weight

        # Apply gradients to driver model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if name in avg_grads:
                param.grad = torch.from_numpy(avg_grads[name])
        optimizer.step()

        # Cleanup broadcast
        model_state_bc.unpersist()
        config_bc.unpersist()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _distributed_evaluate_gpu(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        data_type: str,
        batch_size: int,
    ) -> tuple:
        """
        Distributed GPU inference using Spark executors.

        Each executor loads model onto its GPU and runs inference on its partition.
        """
        sc = self.spark.sparkContext
        num_partitions = max(2, sc.defaultParallelism)

        model_state_bytes = pickle.dumps(model.state_dict())
        model_state_bc = sc.broadcast(model_state_bytes)

        # Partition data (maintain order)
        n_samples = len(X)
        chunk_size = (n_samples + num_partitions - 1) // num_partitions
        data_chunks = []
        for i in range(num_partitions):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            data_chunks.append((i, X[start:end], y[start:end]))

        data_rdd = sc.parallelize(data_chunks, numSlices=num_partitions)

        config = {
            "data_type": data_type,
            "batch_size": batch_size,
            "seed": self.seed,
            "n_features": STRUCTURED_NUM_FEATURES if data_type == "structured" else None,
            "n_classes": STRUCTURED_NUM_CLASSES if data_type == "structured" else UNSTRUCTURED_NUM_CLASSES,
            "image_size": None if data_type == "structured" else UNSTRUCTURED_IMAGE_SIZE,
        }
        config_bc = sc.broadcast(config)

        def predict_on_gpu(data_chunk):
            """Worker function: run GPU inference on a data partition."""
            import torch
            import torch.nn as nn
            import numpy as np

            partition_id, X_local, y_local = data_chunk
            cfg = config_bc.value
            state_bytes = model_state_bc.value

            # GPU assignment
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                device_id = partition_id % gpu_count
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.manual_seed_all(cfg["seed"])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                device = torch.device("cpu")

            torch.manual_seed(cfg["seed"])

            # Recreate model on GPU
            if cfg["data_type"] == "structured":
                local_model = TabularNet(
                    n_features=cfg["n_features"],
                    n_classes=cfg["n_classes"],
                )
            else:
                local_model = ImageCNN(
                    image_size=cfg["image_size"],
                    n_classes=cfg["n_classes"],
                )

            local_model.load_state_dict(pickle.loads(state_bytes))
            local_model = local_model.to(device)
            local_model.eval()

            criterion = nn.CrossEntropyLoss()
            X_tensor = torch.from_numpy(X_local).to(device)
            y_tensor = torch.from_numpy(y_local).to(device)

            bs = cfg["batch_size"]
            n = len(X_local)

            all_preds = []
            all_probs = []
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for start_idx in range(0, n, bs):
                    end_idx = min(start_idx + bs, n)
                    bx = X_tensor[start_idx:end_idx]
                    by = y_tensor[start_idx:end_idx]

                    outputs = local_model(bx)
                    loss = criterion(outputs, by)

                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    total_loss += loss.item() * (end_idx - start_idx)
                    total += (end_idx - start_idx)
                    correct += predicted.eq(by).sum().item()

                    # Move results to CPU for collection
                    all_preds.append(predicted.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

            preds = np.concatenate(all_preds)
            probs_arr = np.concatenate(all_probs)

            # Cleanup GPU
            if device.type == "cuda":
                del local_model, X_tensor, y_tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

            return (partition_id, preds, probs_arr, total_loss, correct, total)

        # Execute and collect ordered results
        partition_results = data_rdd.map(predict_on_gpu).collect()
        partition_results.sort(key=lambda x: x[0])

        # Aggregate
        all_preds = np.concatenate([r[1] for r in partition_results])
        all_probs = np.concatenate([r[2] for r in partition_results])
        total_loss = sum(r[3] for r in partition_results)
        total_correct = sum(r[4] for r in partition_results)
        total_samples = sum(r[5] for r in partition_results)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Cleanup
        model_state_bc.unpersist()
        config_bc.unpersist()

        return avg_loss, accuracy, all_preds, all_probs

    # ------------------------------------------------------------------
    # Resource & metrics collection
    # ------------------------------------------------------------------

    def _collect_spark_metrics(self) -> Dict[str, Any]:
        """Collect Spark-level metrics."""
        sc = self.spark.sparkContext
        metrics = {
            "app_id": sc.applicationId,
            "default_parallelism": sc.defaultParallelism,
            "master": self.spark_master,
        }

        try:
            status_tracker = sc.statusTracker()
            executor_ids = status_tracker.getExecutorInfos()
            metrics["active_executors"] = len(executor_ids) if executor_ids else 0
        except Exception:
            metrics["active_executors"] = -1

        metrics["driver_memory_configured"] = self.driver_memory
        metrics["executor_memory_configured"] = self.executor_memory
        metrics["executor_cores_configured"] = self.executor_cores
        metrics["gpus_per_executor"] = self.gpus_per_executor

        return metrics

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory stats from the driver process."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "free_mb": (
                torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            ) / (1024**2),
            "device_count": torch.cuda.device_count(),
        }

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Hash model state dict."""
        import hashlib
        h = hashlib.sha256()
        for key in sorted(model.state_dict().keys()):
            param = model.state_dict()[key].cpu().numpy()
            h.update(param.tobytes())
        return h.hexdigest()[:32]

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get driver process memory usage."""
        process = psutil.Process()
        mem = process.memory_info()
        return {
            "rss_mb": mem.rss / (1024**2),
            "vms_mb": mem.vms / (1024**2),
            "percent": process.memory_percent(),
        }

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get GC statistics for the driver."""
        gc.collect()
        stats = gc.get_stats()
        return {
            "collections": [s["collections"] for s in stats],
            "total_collected": sum(s["collected"] for s in stats),
            "uncollectable": sum(s["uncollectable"] for s in stats),
        }

    def _get_io_stats(self) -> Dict[str, int]:
        """Get IO counters for the driver process."""
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
        """Collect system + Spark + GPU information."""
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_mb": torch.cuda.get_device_properties(i).total_memory / (1024**2),
                "capability": torch.cuda.get_device_capability(i),
            })

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
            },
            "spark": {
                "version": self.spark.version if self.spark else "N/A",
                "master": self.spark_master,
                "driver_memory": self.driver_memory,
                "executor_memory": self.executor_memory,
                "executor_cores": self.executor_cores,
                "num_executors": self.num_executors,
                "gpus_per_executor": self.gpus_per_executor,
            },
            "gpus": gpu_info,
            "device": "cuda (distributed)",
            "mode": self.mode,
        }
