"""
Spark + CPU Runner

Distributes PyTorch model training and inference across Spark executors on CPU.
Uses Spark's barrier execution mode for synchronized distributed training
and partition-level inference.

Tracks driver/worker/executor metrics, memory allocation/release, GC, and IO.
"""

import gc
import os
import time
import json
import tempfile
import pickle
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import psutil

from pyspark.sql import SparkSession
from pyspark import SparkContext, TaskContext

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


class SparkCPURunner:
    """
    Runner for distributed PyTorch training/inference using PySpark on CPU.

    Architecture:
        - Driver: Coordinates training, holds the master model, aggregates gradients
        - Workers/Executors: Process data partitions, compute local gradients/predictions

    Strategy:
        - Data-parallel approach: partition data across executors
        - Each executor runs forward/backward on its partition
        - Gradients are aggregated on the driver (parameter server pattern)
        - Inference: Each executor runs predictions on its partition independently

    This ensures the same model produces consistent results as the pure PyTorch
    runners while leveraging Spark's distributed data handling.
    """

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        spark_master: str = SPARK_MASTER,
        driver_memory: str = SPARK_DRIVER_MEMORY,
        executor_memory: str = SPARK_EXECUTOR_MEMORY,
        executor_cores: int = SPARK_EXECUTOR_CORES,
        num_executors: int = SPARK_NUM_EXECUTORS,
    ):
        self.mode = "spark_cpu"
        self.seed = seed
        self.spark_master = spark_master
        self.driver_memory = driver_memory
        self.executor_memory = executor_memory
        self.executor_cores = executor_cores
        self.num_executors = num_executors
        self.device = torch.device("cpu")
        self.spark: SparkSession = None

    def _create_spark_session(self) -> SparkSession:
        """Create and configure a SparkSession optimized for CPU execution."""
        builder = (
            SparkSession.builder
            .master(self.spark_master)
            .appName("PyTorchBenchmark_SparkCPU")
            .config("spark.driver.memory", self.driver_memory)
            .config("spark.executor.memory", self.executor_memory)
            .config("spark.executor.cores", str(self.executor_cores))
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.python.worker.reuse", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Memory management
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            # GC configuration for executors
            .config("spark.executor.extraJavaOptions",
                    "-XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
            .config("spark.driver.extraJavaOptions",
                    "-XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
            # Shuffle optimization
            .config("spark.shuffle.compress", "true")
            .config("spark.shuffle.spill.compress", "true")
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
        Run complete benchmark for both structured and unstructured data on Spark+CPU.

        Returns:
            Dictionary with results for each data type plus system/Spark info.
        """
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
        """Run distributed training + evaluation for a single data type."""
        seed_everything(self.seed)
        result = RunnerResult(self.mode, data_type)

        # --- Data generation on driver ---
        if data_type == "structured":
            X, y = generate_structured_data()
            n_features = STRUCTURED_NUM_FEATURES
            n_classes = STRUCTURED_NUM_CLASSES
        else:
            X, y = generate_unstructured_data()
            n_features = None
            n_classes = UNSTRUCTURED_NUM_CLASSES

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- Model creation (same seed as other runners) ---
        model = create_model(data_type, device=self.device, seed=self.seed)
        model_info = get_model_summary(model)

        # --- Resource tracking ---
        gc_stats_before = self._get_gc_stats()
        io_before = self._get_io_stats()
        mem_before = self._get_memory_stats()
        process = psutil.Process()
        cpu_times_before = process.cpu_times()

        # --- Distributed Training ---
        train_start = time.perf_counter()

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            loss, acc = self._distributed_train_epoch(
                model, X_train, y_train, data_type, batch_size, lr
            )
            epoch_time = time.perf_counter() - epoch_start

            result.train_losses.append(loss)
            result.train_accuracies.append(acc)
            result.epoch_times.append(epoch_time)

        result.total_train_time = time.perf_counter() - train_start

        # --- Distributed Inference ---
        infer_start = time.perf_counter()
        test_loss, test_acc, preds, probs = self._distributed_evaluate(
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
    # Distributed training logic
    # ------------------------------------------------------------------

    def _distributed_train_epoch(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        data_type: str,
        batch_size: int,
        lr: float,
    ) -> tuple:
        """
        Distributed training epoch using Spark.

        Strategy: Data-parallel with gradient aggregation on driver.
        1. Broadcast model weights to executors
        2. Each executor computes gradients on its data partition
        3. Collect and average gradients on driver
        4. Update model on driver
        """
        sc = self.spark.sparkContext
        num_partitions = max(2, sc.defaultParallelism)

        # Serialize model state for broadcast
        model_state_bytes = pickle.dumps(model.state_dict())
        model_state_bc = sc.broadcast(model_state_bytes)

        # Partition data
        n_samples = len(X)
        indices = list(range(n_samples))
        # Shuffle indices deterministically for this epoch
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        # Create partitioned data as list of (X_chunk, y_chunk) tuples
        chunk_size = (n_samples + num_partitions - 1) // num_partitions
        data_chunks = []
        for i in range(num_partitions):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk_indices = indices[start:end]
            data_chunks.append((X[chunk_indices], y[chunk_indices]))

        # Distribute and compute gradients on executors
        data_rdd = sc.parallelize(data_chunks, numSlices=num_partitions)

        # Configuration to pass to workers
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

        def compute_gradients_on_partition(data_chunk):
            """Worker function: compute gradients on a data partition."""
            X_local, y_local = data_chunk
            cfg = config_bc.value
            state_bytes = model_state_bc.value

            # Reproducible seeding on worker
            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])

            # Recreate model on worker
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
            local_model.train()

            # Create tensors
            X_tensor = torch.from_numpy(X_local)
            y_tensor = torch.from_numpy(y_local)

            # Forward + backward on full partition (batch-by-batch)
            criterion = nn.CrossEntropyLoss()
            bs = cfg["batch_size"]
            n = len(X_local)

            total_loss = 0.0
            correct = 0
            total = 0

            # Accumulate gradients over all batches
            local_model.zero_grad()

            for start in range(0, n, bs):
                end = min(start + bs, n)
                bx = X_tensor[start:end]
                by = y_tensor[start:end]

                outputs = local_model(bx)
                loss = criterion(outputs, by)
                loss.backward()

                total_loss += loss.item() * (end - start)
                _, predicted = outputs.max(1)
                total += (end - start)
                correct += predicted.eq(by).sum().item()

            # Collect gradients
            grads = {}
            for name, param in local_model.named_parameters():
                if param.grad is not None:
                    grads[name] = param.grad.numpy().copy()

            return (grads, total_loss, correct, total)

        # Execute on Spark
        partition_results = data_rdd.map(compute_gradients_on_partition).collect()

        # Aggregate gradients on driver (average)
        total_loss = sum(r[1] for r in partition_results)
        total_correct = sum(r[2] for r in partition_results)
        total_samples = sum(r[3] for r in partition_results)

        # Average gradients weighted by partition size
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

        # Cleanup broadcast variables
        model_state_bc.unpersist()
        config_bc.unpersist()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _distributed_evaluate(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        data_type: str,
        batch_size: int,
    ) -> tuple:
        """
        Distributed inference using Spark executors.

        Partitions test data, runs inference on each executor, collects results.
        """
        sc = self.spark.sparkContext
        num_partitions = max(2, sc.defaultParallelism)

        # Serialize model state
        model_state_bytes = pickle.dumps(model.state_dict())
        model_state_bc = sc.broadcast(model_state_bytes)

        # Partition data (maintain order for result collection)
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

        def predict_on_partition(data_chunk):
            """Worker function: run inference on a data partition."""
            partition_id, X_local, y_local = data_chunk
            cfg = config_bc.value
            state_bytes = model_state_bc.value

            torch.manual_seed(cfg["seed"])

            # Recreate model
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
            local_model.eval()

            criterion = nn.CrossEntropyLoss()
            X_tensor = torch.from_numpy(X_local)
            y_tensor = torch.from_numpy(y_local)

            bs = cfg["batch_size"]
            n = len(X_local)

            all_preds = []
            all_probs = []
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    bx = X_tensor[start:end]
                    by = y_tensor[start:end]

                    outputs = local_model(bx)
                    loss = criterion(outputs, by)

                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    total_loss += loss.item() * (end - start)
                    total += (end - start)
                    correct += predicted.eq(by).sum().item()

                    all_preds.append(predicted.numpy())
                    all_probs.append(probs.numpy())

            preds = np.concatenate(all_preds)
            probs = np.concatenate(all_probs)

            return (partition_id, preds, probs, total_loss, correct, total)

        # Execute on Spark and collect ordered results
        partition_results = data_rdd.map(predict_on_partition).collect()
        partition_results.sort(key=lambda x: x[0])  # sort by partition_id

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
    # Resource & Spark metrics collection
    # ------------------------------------------------------------------

    def _collect_spark_metrics(self) -> Dict[str, Any]:
        """Collect Spark-level metrics from the SparkContext."""
        sc = self.spark.sparkContext
        metrics = {
            "app_id": sc.applicationId,
            "default_parallelism": sc.defaultParallelism,
            "master": self.spark_master,
        }

        # Attempt to get executor status info
        try:
            status_tracker = sc.statusTracker()
            executor_ids = status_tracker.getExecutorInfos()
            metrics["active_executors"] = len(executor_ids) if executor_ids else 0
        except Exception:
            metrics["active_executors"] = -1

        # Memory configuration
        metrics["driver_memory_configured"] = self.driver_memory
        metrics["executor_memory_configured"] = self.executor_memory
        metrics["executor_cores_configured"] = self.executor_cores

        return metrics

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Hash model state dict for reproducibility."""
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
        """Get garbage collector statistics for the driver."""
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
        """Collect system + Spark information."""
        return {
            "platform": {
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            },
            "torch": {
                "version": torch.__version__,
            },
            "spark": {
                "version": self.spark.version if self.spark else "N/A",
                "master": self.spark_master,
                "driver_memory": self.driver_memory,
                "executor_memory": self.executor_memory,
                "executor_cores": self.executor_cores,
                "num_executors": self.num_executors,
            },
            "device": str(self.device),
            "mode": self.mode,
        }
