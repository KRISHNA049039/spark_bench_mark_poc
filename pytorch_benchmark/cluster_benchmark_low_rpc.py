"""
Low-RPC Cluster Benchmark — Minimize Network Overhead

Strategy: Instead of sending model weights + data over Spark RPC,
workers generate data locally and download model weights once from
a shared location. Only tiny results (hashes, metrics) travel over RPC.

Approaches:
1. Workers download model from torchvision cache (not serialized over Spark)
2. Workers generate input data locally from seed (not transferred)
3. Only prediction hashes + timing metrics returned (few KB, fits in RPC)
4. Single SparkSession reused across all phases (no repeated session overhead)

Network transfer comparison:
  Before: 73 MB per task × 8 tasks = 584 MB over Spark RPC
  After:  ~1 KB config per task × 8 tasks = 8 KB over Spark RPC

Usage:
    set SPARK_MASTER=spark://192.168.4.100:7077
    set BENCHMARK_MODELS=efficientnet_b0
    python -m pytorch_benchmark.cluster_benchmark_low_rpc
"""

import gc
import hashlib
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cluster_low_rpc")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_DRIVER_HOST = os.environ.get("SPARK_DRIVER_HOST", "")
SPARK_DRIVER_PORT = os.environ.get("SPARK_DRIVER_PORT", "")
SPARK_DRIVER_BLOCKMANAGER_PORT = os.environ.get("SPARK_DRIVER_BLOCKMANAGER_PORT", "")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
SPARK_EXECUTOR_CORES = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))

NUM_SAMPLES = int(os.environ.get("BENCHMARK_SAMPLES", "200"))
BATCH_SIZE = int(os.environ.get("BENCHMARK_BATCH_SIZE", "64"))
NUM_PARTITIONS = int(os.environ.get("BENCHMARK_PARTITIONS", "4"))
OUTPUT_DIR = os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_results")
SEED = 42
FORCE_GPU = os.environ.get("FORCE_GPU_PHASES", "true").lower() in ("true", "1", "yes")

ALL_MODELS = os.environ.get(
    "BENCHMARK_MODELS",
    "resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep"
).split(",")


# ---------------------------------------------------------------------------
# Spark session (reused across all phases — no repeated session overhead)
# ---------------------------------------------------------------------------

_spark_session = None


def get_spark():
    """Get or create a single SparkSession (reused across all models/phases)."""
    global _spark_session
    
    # Check if existing session is still alive
    if _spark_session is not None:
        try:
            # Test if session is still valid
            _spark_session.sparkContext._jsc.sc().isStopped()
            if not _spark_session.sparkContext._jsc.sc().isStopped():
                return _spark_session
        except Exception:
            pass
        # Session is dead, clear it
        _spark_session = None

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName("LowRPC_Benchmark")
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.executor.cores", str(SPARK_EXECUTOR_CORES))
        .config("spark.driver.memory", "4g")
        .config("spark.python.worker.reuse", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Reduce shuffle overhead
        .config("spark.shuffle.compress", "true")
        .config("spark.io.compression.codec", "zstd")
        # Increase RPC message size (so results come via RPC, not block transfer)
        .config("spark.rpc.message.maxSize", "256")
        # Timeout tuning
        .config("spark.network.timeout", "300s")
        .config("spark.executor.heartbeatInterval", "30s")
        # Disable dynamic allocation (predictable behavior)
        .config("spark.dynamicAllocation.enabled", "false")
        # Pass CUDA env to executor processes
        .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "0")
        .config("spark.executorEnv.NVIDIA_VISIBLE_DEVICES", "all")
    )

    if SPARK_DRIVER_HOST:
        builder = builder.config("spark.driver.host", SPARK_DRIVER_HOST)
        builder = builder.config("spark.driver.bindAddress", "0.0.0.0")
    if SPARK_DRIVER_PORT:
        builder = builder.config("spark.driver.port", SPARK_DRIVER_PORT)
    if SPARK_DRIVER_BLOCKMANAGER_PORT:
        builder = builder.config("spark.driver.blockManager.port", SPARK_DRIVER_BLOCKMANAGER_PORT)

    _spark_session = builder.getOrCreate()
    return _spark_session


def stop_spark():
    """Stop the shared SparkSession."""
    global _spark_session
    if _spark_session:
        _spark_session.stop()
        _spark_session = None


# ---------------------------------------------------------------------------
# Worker function — SELF-CONTAINED (no large data sent over RPC)
# ---------------------------------------------------------------------------

def worker_inference(config_tuple):
    """
    Self-contained worker function.

    NOTHING large is sent over Spark RPC. The worker:
    1. Receives only a tiny config dict (model name, seed, partition info)
    2. Downloads/loads model locally (from torchvision cache or generates weights)
    3. Generates input data locally from seed
    4. Runs inference
    5. Returns ONLY: hash + timing + memory metrics (few KB)

    This eliminates:
    - 73 MB model serialization per task
    - 75 MB input data transfer per task
    - Block transfer issues (results are tiny, fit in RPC)
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import time as _time
    import gc as _gc
    import hashlib as _hashlib
    import os as _os

    # Unpack tiny config (< 1 KB)
    (partition_id, num_partitions, model_name, num_samples,
     batch_size, seed, phase, total_partitions) = config_tuple

    # ---- Determine device ----
    # Force CUDA visibility in worker process
    if "CUDA_VISIBLE_DEVICES" not in _os.environ:
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialize CUDA in this worker process.
    # IMPORTANT: don't silently swallow the reason CUDA wasn't available --
    # capture it so it comes back to the driver in the result dict instead
    # of disappearing into a worker-side log you may never see.
    cuda_available = False
    cuda_diagnostic = "not attempted"
    try:
        import socket
        import sys as _sys
        cuda_diagnostic = (
            f"host={socket.gethostname()} python={_sys.executable} "
            f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
        )
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.init()
            cuda_diagnostic += f" device_count={torch.cuda.device_count()}"
        else:
            reason = (
                "torch is a CPU-only build (torch.version.cuda is None)"
                if torch.version.cuda is None
                else "torch.cuda.is_available() returned False "
                     "(driver/GPU not visible to this process)"
            )
            cuda_diagnostic += f" reason={reason}"
    except Exception as e:
        cuda_available = False
        cuda_diagnostic += f" EXCEPTION={type(e).__name__}: {e}"

    if phase == "gpu" or (phase == "hybrid" and partition_id % 2 == 0):
        if cuda_available:
            device = torch.device(f"cuda:{partition_id % torch.cuda.device_count()}")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load model LOCALLY (no RPC transfer) ----
    # torchvision models download weights to local cache (~/.cache/torch)
    # This happens ONCE per worker, then cached on disk
    import torchvision.models as models

    model_loaders = {
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "mobilenet_v3": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    }

    # For non-torchvision models, create locally with fixed seed
    if model_name in model_loaders:
        model = model_loaders[model_name]()
    elif model_name == "distilbert":
        # Recreate DistilBERT architecture locally
        from pytorch_benchmark.pretrained_models import DistilBERTClassifier
        torch.manual_seed(seed)
        model = DistilBERTClassifier()
    elif model_name == "tabular_deep":
        from pytorch_benchmark.pretrained_models import DeepTabularModel
        torch.manual_seed(seed)
        model = DeepTabularModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()

    # ---- Generate input data LOCALLY from seed (no transfer) ----
    # Each partition generates its own slice deterministically
    rng = np.random.RandomState(seed + partition_id)

    samples_per_partition = num_samples // total_partitions
    extra = num_samples % total_partitions
    if partition_id < extra:
        samples_per_partition += 1

    # Generate appropriate input shape
    from pytorch_benchmark.pretrained_models import AVAILABLE_MODELS
    model_config = AVAILABLE_MODELS[model_name]

    if model_config["type"] == "vision":
        C, H, W = model_config["input_size"]
        data = rng.randn(samples_per_partition, C, H, W).astype(np.float32)
    elif model_config["type"] == "nlp":
        seq_len = model_config["max_seq_length"]
        data = rng.randint(0, 30522, size=(samples_per_partition, seq_len))
        data = data.astype(np.int64)
    elif model_config["type"] == "tabular":
        n_feat = model_config["num_features"]
        data = rng.randn(samples_per_partition, n_feat).astype(np.float32)

    # ---- Run inference ----
    import psutil as _psutil
    process = _psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    input_tensor = torch.from_numpy(data).to(device)
    all_preds = []

    exec_start = _time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = input_tensor[i:min(i + batch_size, len(data))]
            output = model(batch)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    exec_time = _time.perf_counter() - exec_start
    predictions = np.concatenate(all_preds)

    # ---- Compute hash ----
    pred_hash = _hashlib.sha256(predictions.tobytes()).hexdigest()[:16]

    # ---- Collect metrics ----
    mem_after = process.memory_info().rss / (1024**2)

    gpu_stats = {}
    if device.type == "cuda":
        gpu_stats = {
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "device_name": torch.cuda.get_device_name(device),
        }
        del model, input_tensor
        torch.cuda.empty_cache()
    else:
        del model, input_tensor

    _gc.collect()

    # ---- Return ONLY tiny result (< 1 KB) ----
    # No large arrays, no model state — just metrics
    return {
        "partition_id": partition_id,
        "predictions_hash": pred_hash,
        "num_samples": samples_per_partition,
        "device": str(device),
        "execution_time_sec": exec_time,
        "throughput": samples_per_partition / exec_time,
        "memory_delta_mb": mem_after - mem_before,
        "gpu": gpu_stats,
        "cuda_diagnostic": cuda_diagnostic,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_baseline(model_name: str) -> Dict[str, Any]:
    """Local CPU baseline (no Spark)."""
    from pytorch_benchmark.pretrained_models import (
        load_pretrained_model, generate_vision_inference_data,
        generate_nlp_inference_data, generate_tabular_inference_data,
        AVAILABLE_MODELS,
    )
    from pytorch_benchmark.data_generation import seed_everything

    seed_everything(SEED)
    model_config = AVAILABLE_MODELS[model_name]

    # Generate data
    if model_config["type"] == "vision":
        input_data = generate_vision_inference_data(NUM_SAMPLES, model_config["input_size"], SEED)
    elif model_config["type"] == "nlp":
        input_data = generate_nlp_inference_data(NUM_SAMPLES, model_config["max_seq_length"], seed=SEED)
    elif model_config["type"] == "tabular":
        input_data = generate_tabular_inference_data(NUM_SAMPLES, model_config["num_features"], SEED)

    model, info = load_pretrained_model(model_name, device=torch.device("cpu"))
    model.eval()
    all_preds = []

    # Warmup
    _ = model(input_data[:BATCH_SIZE])

    start = time.perf_counter()
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch = input_data[i:min(i + BATCH_SIZE, NUM_SAMPLES)]
        out = model(batch)
        all_preds.append(out.argmax(dim=1).numpy())
    elapsed = time.perf_counter() - start

    preds = np.concatenate(all_preds)
    pred_hash = hashlib.sha256(preds.tobytes()).hexdigest()[:16]

    del model
    gc.collect()

    return {
        "throughput_samples_per_sec": NUM_SAMPLES / elapsed,
        "total_time_sec": elapsed,
        "predictions_hash": pred_hash,
        "num_samples": NUM_SAMPLES,
    }


def run_distributed_phase(model_name: str, phase: str) -> Dict[str, Any]:
    """
    Run distributed phase with MINIMAL RPC overhead.

    Each task receives only ~200 bytes of config.
    Workers load model and generate data locally.
    Workers return only ~500 bytes of metrics.
    """
    spark = get_spark()
    sc = spark.sparkContext

    # Create tiny config tuples (< 200 bytes each, NO large data)
    configs = [
        (
            partition_id,       # which partition
            NUM_PARTITIONS,     # total partitions
            model_name,         # model to load locally
            NUM_SAMPLES,        # total samples
            BATCH_SIZE,         # batch size
            SEED,               # seed for reproducibility
            phase,              # "cpu", "gpu", or "hybrid"
            NUM_PARTITIONS,     # for sample count calculation
        )
        for partition_id in range(NUM_PARTITIONS)
    ]

    # Distribute — each task is ~200 bytes (not 73 MB!)
    config_rdd = sc.parallelize(configs, numSlices=NUM_PARTITIONS)

    # Execute on workers
    infer_start = time.perf_counter()
    results = config_rdd.map(worker_inference).collect()
    total_time = time.perf_counter() - infer_start

    # Aggregate results (already sorted by partition_id from map)
    results.sort(key=lambda x: x["partition_id"])

    # Print per-partition device + CUDA diagnostic immediately, so a GPU
    # phase that's silently running on CPU on some/all nodes is obvious
    # right here instead of buried in the JSON file.
    if phase in ("gpu", "hybrid"):
        for r in results:
            logger.info(
                f"    partition={r['partition_id']} device={r['device']} "
                f"| {r.get('cuda_diagnostic', 'n/a')}"
            )

    # Combine hashes deterministically
    combined_hash = hashlib.sha256(
        "".join(r["predictions_hash"] for r in results).encode()
    ).hexdigest()[:16]

    # Aggregate metrics
    exec_times = [r["execution_time_sec"] for r in results]
    throughputs = [r["throughput"] for r in results]
    total_samples = sum(r["num_samples"] for r in results)

    return {
        "phase": phase,
        "throughput_samples_per_sec": total_samples / total_time,
        "total_time_sec": total_time,
        "predictions_hash": combined_hash,
        "num_samples": total_samples,
        "num_partitions": NUM_PARTITIONS,
        "executor_metrics": {
            "num_executors_used": len(results),
            "avg_exec_time_sec": np.mean(exec_times),
            "max_exec_time_sec": np.max(exec_times),
            "min_exec_time_sec": np.min(exec_times),
            "total_throughput": sum(throughputs),
            "devices_used": list(set(r["device"] for r in results)),
            "per_partition": results,
        },
        "rpc_overhead": {
            "bytes_sent_per_task": "~200 bytes (config only)",
            "bytes_received_per_task": "~500 bytes (metrics only)",
            "model_transferred": False,
            "data_transferred": False,
            "strategy": "Workers load model + generate data locally",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("LOW-RPC CLUSTER BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Spark Master: {SPARK_MASTER}")
    logger.info(f"Samples: {NUM_SAMPLES}, Batch: {BATCH_SIZE}, Partitions: {NUM_PARTITIONS}")
    logger.info(f"Models: {ALL_MODELS}")
    logger.info(f"Strategy: Workers load model + data locally. Only config/metrics over RPC.")
    logger.info("")
    logger.info("RPC transfer per task:")
    logger.info("  SENT:     ~200 bytes (model name + seed + partition config)")
    logger.info("  RECEIVED: ~500 bytes (hash + timing + memory metrics)")
    logger.info("  vs before: 73 MB sent + results via block transfer")
    logger.info("")

    all_results = {}

    for model_name in ALL_MODELS:
        logger.info(f"\n{'━' * 70}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'━' * 70}")

        model_results = {}

        # Baseline
        logger.info("\n  [BASELINE] Local CPU...")
        baseline = run_baseline(model_name)
        model_results["baseline_cpu"] = baseline
        logger.info(f"  [BASELINE] {baseline['throughput_samples_per_sec']:.1f} s/s | hash={baseline['predictions_hash']}")

        # Phase 1: Distributed CPU
        logger.info(f"\n  [PHASE 1] Distributed CPU ({NUM_PARTITIONS} partitions, LOW RPC)...")
        phase1 = run_distributed_phase(model_name, "cpu")
        model_results["phase1_dist_cpu"] = phase1
        if "error" not in phase1:
            logger.info(f"  [PHASE 1] {phase1['throughput_samples_per_sec']:.1f} s/s | hash={phase1['predictions_hash']} | time={phase1['total_time_sec']:.2f}s")
        else:
            logger.warning(f"  [PHASE 1] FAILED: {phase1.get('error', '')[:80]}")

        # Phase 2: Distributed GPU
        if FORCE_GPU or torch.cuda.is_available():
            logger.info(f"\n  [PHASE 2] Distributed GPU ({NUM_PARTITIONS} partitions, LOW RPC)...")
            phase2 = run_distributed_phase(model_name, "gpu")
            model_results["phase2_dist_gpu"] = phase2
            if "error" not in phase2:
                logger.info(f"  [PHASE 2] {phase2['throughput_samples_per_sec']:.1f} s/s | hash={phase2['predictions_hash']} | time={phase2['total_time_sec']:.2f}s")
            else:
                logger.warning(f"  [PHASE 2] FAILED: {phase2.get('error', '')[:80]}")

        # Phase 3: Hybrid
        if FORCE_GPU or torch.cuda.is_available():
            logger.info(f"\n  [PHASE 3] Hybrid CPU+GPU ({NUM_PARTITIONS} partitions, LOW RPC)...")
            phase3 = run_distributed_phase(model_name, "hybrid")
            model_results["phase3_hybrid"] = phase3
            if "error" not in phase3:
                logger.info(f"  [PHASE 3] {phase3['throughput_samples_per_sec']:.1f} s/s | hash={phase3['predictions_hash']} | time={phase3['total_time_sec']:.2f}s")
            else:
                logger.warning(f"  [PHASE 3] FAILED: {phase3.get('error', '')[:80]}")

        all_results[model_name] = model_results

    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n{'Model':<16} {'Baseline':<12} {'Dist CPU':<12} {'Dist GPU':<12} {'Hybrid':<12}")
    logger.info("-" * 70)

    for model_name in ALL_MODELS:
        if model_name not in all_results:
            continue
        mr = all_results[model_name]
        bl = mr.get("baseline_cpu", {}).get("throughput_samples_per_sec", 0)
        p1 = mr.get("phase1_dist_cpu", {}).get("throughput_samples_per_sec", 0)
        p2 = mr.get("phase2_dist_gpu", {}).get("throughput_samples_per_sec", 0)
        p3 = mr.get("phase3_hybrid", {}).get("throughput_samples_per_sec", 0)
        logger.info(f"{model_name:<16} {bl:<12.1f} {p1:<12.1f} {p2:<12.1f} {p3:<12.1f}")

    # Save
    stop_spark()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"cluster_low_rpc_{timestamp}.json")

    def serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=serializer)
    logger.info(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
