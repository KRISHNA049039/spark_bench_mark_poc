"""
3-Phase Cluster Benchmark Runner

Runs inference on all models sequentially across 3 phases:
  Phase 1: Distributed CPU — all cores, no GPU, max CPU parallelism
  Phase 2: Distributed GPU — GPU executors only, minimize CPU usage
  Phase 3: Hybrid CPU+GPU — both CPU and GPU working simultaneously

Each phase runs models one-by-one to isolate performance characteristics.
Results are collected and compared at the end with a delta analysis.

Usage:
    python -m pytorch_benchmark.cluster_benchmark

Environment variables (set in docker-compose):
    SPARK_MASTER            — spark://host:port
    SPARK_DRIVER_HOST       — driver's advertised hostname
    SPARK_DRIVER_PORT       — driver's port for executor callbacks
    SPARK_EXECUTOR_MEMORY   — heap per executor
    BENCHMARK_SAMPLES       — number of inference samples (default: 1000)
    BENCHMARK_BATCH_SIZE    — batch size (default: 64)
    BENCHMARK_PARTITIONS    — number of data partitions (default: 8)
"""

import gc
import hashlib
import json
import os
import pickle
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import psutil

from pytorch_benchmark.pretrained_models import (
    load_pretrained_model,
    generate_vision_inference_data,
    generate_nlp_inference_data,
    generate_tabular_inference_data,
    AVAILABLE_MODELS,
)
from pytorch_benchmark.data_generation import seed_everything

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_DRIVER_HOST = os.environ.get("SPARK_DRIVER_HOST", "")
SPARK_DRIVER_PORT = os.environ.get("SPARK_DRIVER_PORT", "")
SPARK_DRIVER_BLOCKMANAGER_PORT = os.environ.get("SPARK_DRIVER_BLOCKMANAGER_PORT", "")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "12g")
SPARK_EXECUTOR_MEMORY_OVERHEAD = os.environ.get("SPARK_EXECUTOR_MEMORY_OVERHEAD", "2g")
SPARK_EXECUTOR_CORES = int(os.environ.get("SPARK_EXECUTOR_CORES", "4"))
SPARK_NUM_EXECUTORS = int(os.environ.get("SPARK_NUM_EXECUTORS", "4"))

NUM_SAMPLES = int(os.environ.get("BENCHMARK_SAMPLES", "1000"))
BATCH_SIZE = int(os.environ.get("BENCHMARK_BATCH_SIZE", "64"))
NUM_PARTITIONS = int(os.environ.get("BENCHMARK_PARTITIONS", "8"))
OUTPUT_DIR = os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_results")
SEED = 42

ALL_MODELS = ["resnet50", "mobilenet_v3", "efficientnet_b0", "distilbert", "tabular_deep"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cluster_benchmark")


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str, use_gpu: bool = False):
    """Create an optimized SparkSession for the cluster."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(app_name)
        # Memory
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.executor.memoryOverhead", SPARK_EXECUTOR_MEMORY_OVERHEAD)
        .config("spark.executor.cores", str(SPARK_EXECUTOR_CORES))
        .config("spark.driver.memory", "4g")
        # Parallelism
        .config("spark.default.parallelism", str(NUM_PARTITIONS))
        .config("spark.sql.shuffle.partitions", str(NUM_PARTITIONS))
        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.python.worker.reuse", "true")
        # Network
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        # Memory management
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.2")
        # GC optimization
        .config("spark.executor.extraJavaOptions",
                "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 "
                "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
    )

    # Driver host for cluster mode (remote executors connect back)
    if SPARK_DRIVER_HOST:
        builder = builder.config("spark.driver.host", SPARK_DRIVER_HOST)
        builder = builder.config("spark.driver.bindAddress", "0.0.0.0")
    if SPARK_DRIVER_PORT:
        builder = builder.config("spark.driver.port", SPARK_DRIVER_PORT)
    if SPARK_DRIVER_BLOCKMANAGER_PORT:
        builder = builder.config("spark.driver.blockManager.port", SPARK_DRIVER_BLOCKMANAGER_PORT)

    # GPU resources
    if use_gpu:
        builder = (builder
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", "1")
        )

    return builder.getOrCreate()


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def generate_input_data(model_name: str) -> torch.Tensor:
    """Generate input data for a model."""
    config = AVAILABLE_MODELS[model_name]
    seed_everything(SEED)

    if config["type"] == "vision":
        return generate_vision_inference_data(
            num_samples=NUM_SAMPLES, image_size=config["input_size"], seed=SEED
        )
    elif config["type"] == "nlp":
        return generate_nlp_inference_data(
            num_samples=NUM_SAMPLES, max_seq_length=config["max_seq_length"], seed=SEED
        )
    elif config["type"] == "tabular":
        return generate_tabular_inference_data(
            num_samples=NUM_SAMPLES, num_features=config["num_features"], seed=SEED
        )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


@torch.no_grad()
def run_local_cpu_baseline(model_name: str, input_data: torch.Tensor) -> Dict[str, Any]:
    """Run local CPU inference as baseline (no Spark)."""
    seed_everything(SEED)
    model, model_info = load_pretrained_model(model_name, device=torch.device("cpu"))
    model.eval()

    n = len(input_data)
    all_preds = []

    # Warmup
    _ = model(input_data[:BATCH_SIZE])

    start = time.perf_counter()
    for i in range(0, n, BATCH_SIZE):
        batch = input_data[i:min(i + BATCH_SIZE, n)]
        out = model(batch)
        all_preds.append(out.argmax(dim=1).numpy())
    elapsed = time.perf_counter() - start

    preds = np.concatenate(all_preds)
    pred_hash = hashlib.sha256(preds.tobytes()).hexdigest()[:16]

    del model
    gc.collect()

    return {
        "throughput_samples_per_sec": n / elapsed,
        "total_time_sec": elapsed,
        "avg_latency_ms": (elapsed / n) * 1000,
        "predictions_hash": pred_hash,
        "num_samples": n,
        "model_info": model_info,
    }


def run_distributed_phase(
    model_name: str,
    input_data: torch.Tensor,
    phase: str,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Run distributed inference via Spark cluster.

    phase: 'cpu', 'gpu', or 'hybrid'
    """
    spark = None
    try:
        # Load model for serialization
        model, model_info = load_pretrained_model(model_name, device=torch.device("cpu"))
        model_state_bytes = pickle.dumps(model.state_dict())
        del model
        gc.collect()

        # Create Spark session
        spark = create_spark_session(
            app_name=f"Phase_{phase}_{model_name}",
            use_gpu=(use_gpu and phase != "cpu"),
        )
        sc = spark.sparkContext

        # Partition data
        input_np = input_data.numpy()
        n = len(input_np)
        partitions = NUM_PARTITIONS
        chunk_size = (n + partitions - 1) // partitions
        chunks = [
            (i, input_np[i * chunk_size: min((i + 1) * chunk_size, n)])
            for i in range(partitions)
        ]

        data_rdd = sc.parallelize(chunks, numSlices=partitions)
        model_bc = sc.broadcast(model_state_bytes)

        config = {
            "model_name": model_name,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "phase": phase,
            "use_gpu": use_gpu,
        }
        config_bc = sc.broadcast(config)

        # Time the distributed inference
        infer_start = time.perf_counter()

        def infer_on_executor(chunk):
            """Runs on each executor — CPU or GPU depending on phase."""
            import torch
            import numpy as np
            import pickle as pkl
            import time as _time
            import psutil as _psutil
            from pytorch_benchmark.pretrained_models import load_pretrained_model

            partition_id, data = chunk
            cfg = config_bc.value
            state_bytes = model_bc.value

            # Device selection based on phase
            if cfg["phase"] == "gpu" or (cfg["phase"] == "hybrid" and cfg["use_gpu"]):
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    device = torch.device(f"cuda:{partition_id % gpu_count}")
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")

            torch.manual_seed(cfg["seed"])

            # Load model
            model, _ = load_pretrained_model(cfg["model_name"], device=device)
            model.load_state_dict(pkl.loads(state_bytes))
            model.eval()

            # Track resources
            process = _psutil.Process()
            mem_before = process.memory_info().rss / (1024**2)
            exec_start = _time.perf_counter()

            # Run inference
            input_tensor = torch.from_numpy(data).to(device)
            bs = cfg["batch_size"]
            preds_list = []

            with torch.no_grad():
                for s in range(0, len(data), bs):
                    e = min(s + bs, len(data))
                    out = model(input_tensor[s:e])
                    preds_list.append(out.argmax(dim=1).cpu().numpy())

            exec_time = _time.perf_counter() - exec_start
            mem_after = process.memory_info().rss / (1024**2)

            predictions = np.concatenate(preds_list)

            # GPU stats
            gpu_stats = {}
            if device.type == "cuda":
                gpu_stats = {
                    "device": str(device),
                    "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "device_name": torch.cuda.get_device_name(device),
                }
                del model, input_tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            else:
                del model

            executor_metrics = {
                "partition_id": partition_id,
                "device": str(device),
                "samples_processed": len(data),
                "execution_time_sec": exec_time,
                "throughput": len(data) / exec_time,
                "memory_delta_mb": mem_after - mem_before,
                "gpu": gpu_stats,
            }

            return (partition_id, predictions, executor_metrics)

        # For hybrid mode: split partitions between CPU and GPU
        if phase == "hybrid":
            # Half partitions on GPU, half on CPU
            gpu_partitions = partitions // 2
            cpu_partitions = partitions - gpu_partitions

            # Create two configs
            config_gpu = dict(config, use_gpu=True, phase="gpu")
            config_cpu = dict(config, use_gpu=False, phase="cpu")
            config_gpu_bc = sc.broadcast(config_gpu)
            config_cpu_bc = sc.broadcast(config_cpu)

            def infer_hybrid(chunk):
                """Route to GPU or CPU based on partition index."""
                partition_id = chunk[0]
                # Even partitions → GPU, Odd partitions → CPU
                if partition_id % 2 == 0:
                    # Override config for this call
                    import builtins
                    builtins._hybrid_config = config_gpu_bc.value
                else:
                    import builtins
                    builtins._hybrid_config = config_cpu_bc.value

                # Use the standard infer function with modified config
                return infer_on_executor(chunk)

            # For hybrid, we modify the config broadcast
            # Simpler approach: just use GPU for even partitions via the phase check
            config_hybrid = dict(config, phase="hybrid", use_gpu=True)
            config_bc.unpersist()
            config_bc_new = sc.broadcast(config_hybrid)

            # Override: route based on partition_id
            def infer_hybrid_v2(chunk):
                import torch
                import numpy as np
                import pickle as pkl
                import time as _time
                import psutil as _psutil
                from pytorch_benchmark.pretrained_models import load_pretrained_model

                partition_id, data = chunk
                cfg = config_bc_new.value
                state_bytes = model_bc.value

                # Even partitions → GPU, Odd → CPU
                use_device_gpu = (partition_id % 2 == 0) and torch.cuda.is_available()

                if use_device_gpu:
                    gpu_count = torch.cuda.device_count()
                    device = torch.device(f"cuda:{partition_id % gpu_count}")
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                else:
                    device = torch.device("cpu")

                torch.manual_seed(cfg["seed"])
                model, _ = load_pretrained_model(cfg["model_name"], device=device)
                model.load_state_dict(pkl.loads(state_bytes))
                model.eval()

                process = _psutil.Process()
                mem_before = process.memory_info().rss / (1024**2)
                exec_start = _time.perf_counter()

                input_tensor = torch.from_numpy(data).to(device)
                bs = cfg["batch_size"]
                preds_list = []

                with torch.no_grad():
                    for s in range(0, len(data), bs):
                        e = min(s + bs, len(data))
                        out = model(input_tensor[s:e])
                        preds_list.append(out.argmax(dim=1).cpu().numpy())

                exec_time = _time.perf_counter() - exec_start
                mem_after = process.memory_info().rss / (1024**2)
                predictions = np.concatenate(preds_list)

                gpu_stats = {}
                if device.type == "cuda":
                    gpu_stats = {
                        "device": str(device),
                        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                        "device_name": torch.cuda.get_device_name(device),
                    }
                    del model, input_tensor
                    torch.cuda.empty_cache()
                else:
                    del model

                return (partition_id, predictions, {
                    "partition_id": partition_id,
                    "device": str(device),
                    "samples_processed": len(data),
                    "execution_time_sec": exec_time,
                    "throughput": len(data) / exec_time,
                    "memory_delta_mb": mem_after - mem_before,
                    "gpu": gpu_stats,
                })

            results = data_rdd.map(infer_hybrid_v2).collect()
            config_bc_new.unpersist()
        else:
            results = data_rdd.map(infer_on_executor).collect()

        total_time = time.perf_counter() - infer_start

        # Sort by partition_id to maintain order
        results.sort(key=lambda x: x[0])

        # Aggregate
        all_preds = np.concatenate([r[1] for r in results])
        executor_metrics = [r[2] for r in results]
        pred_hash = hashlib.sha256(all_preds.tobytes()).hexdigest()[:16]

        # Compute aggregate stats
        exec_times = [m["execution_time_sec"] for m in executor_metrics]
        exec_throughputs = [m["throughput"] for m in executor_metrics]

        model_bc.unpersist()
        config_bc.unpersist()
        spark.stop()

        return {
            "phase": phase,
            "throughput_samples_per_sec": n / total_time,
            "total_time_sec": total_time,
            "avg_latency_ms": (total_time / n) * 1000,
            "predictions_hash": pred_hash,
            "num_samples": n,
            "num_partitions": partitions,
            "model_info": model_info,
            "executor_metrics": {
                "num_executors_used": len(executor_metrics),
                "avg_exec_time_sec": np.mean(exec_times),
                "max_exec_time_sec": np.max(exec_times),
                "min_exec_time_sec": np.min(exec_times),
                "total_throughput": sum(exec_throughputs),
                "avg_throughput_per_executor": np.mean(exec_throughputs),
                "devices_used": list(set(m["device"] for m in executor_metrics)),
                "per_partition": executor_metrics,
            },
        }

    except Exception as e:
        if spark:
            try:
                spark.stop()
            except Exception:
                pass
        return {"phase": phase, "error": str(e)}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all_phases():
    """Run all 3 phases for all models and collect results."""
    logger.info("=" * 70)
    logger.info("3-PHASE CLUSTER BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Spark Master: {SPARK_MASTER}")
    logger.info(f"Samples: {NUM_SAMPLES}, Batch: {BATCH_SIZE}, Partitions: {NUM_PARTITIONS}")
    logger.info(f"Executor Memory: {SPARK_EXECUTOR_MEMORY} + {SPARK_EXECUTOR_MEMORY_OVERHEAD} overhead")
    logger.info(f"Executor Cores: {SPARK_EXECUTOR_CORES}, Num Executors: {SPARK_NUM_EXECUTORS}")
    logger.info(f"Models: {ALL_MODELS}")
    logger.info("")

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU detected — Phase 2 (GPU) and Phase 3 (Hybrid) will skip GPU")

    all_results = {}

    for model_name in ALL_MODELS:
        logger.info(f"\n{'━' * 70}")
        logger.info(f"MODEL: {model_name} ({AVAILABLE_MODELS[model_name]['description']})")
        logger.info(f"{'━' * 70}")

        # Generate data once per model
        input_data = generate_input_data(model_name)
        model_results = {}

        # --- Baseline: Local CPU (no Spark) ---
        logger.info(f"\n  [BASELINE] Local CPU inference...")
        baseline = run_local_cpu_baseline(model_name, input_data)
        model_results["baseline_cpu"] = baseline
        logger.info(
            f"  [BASELINE] {baseline['throughput_samples_per_sec']:.1f} samples/s | "
            f"hash={baseline['predictions_hash']}"
        )

        # --- Phase 1: Distributed CPU ---
        logger.info(f"\n  [PHASE 1] Distributed CPU ({NUM_PARTITIONS} partitions)...")
        phase1 = run_distributed_phase(model_name, input_data, phase="cpu", use_gpu=False)
        model_results["phase1_dist_cpu"] = phase1
        if "error" not in phase1:
            logger.info(
                f"  [PHASE 1] {phase1['throughput_samples_per_sec']:.1f} samples/s | "
                f"hash={phase1['predictions_hash']}"
            )
        else:
            logger.warning(f"  [PHASE 1] FAILED: {phase1['error'][:80]}")

        # --- Phase 2: Distributed GPU ---
        if has_gpu:
            logger.info(f"\n  [PHASE 2] Distributed GPU ({NUM_PARTITIONS} partitions)...")
            phase2 = run_distributed_phase(model_name, input_data, phase="gpu", use_gpu=True)
            model_results["phase2_dist_gpu"] = phase2
            if "error" not in phase2:
                logger.info(
                    f"  [PHASE 2] {phase2['throughput_samples_per_sec']:.1f} samples/s | "
                    f"hash={phase2['predictions_hash']}"
                )
            else:
                logger.warning(f"  [PHASE 2] FAILED: {phase2['error'][:80]}")
        else:
            model_results["phase2_dist_gpu"] = {"phase": "gpu", "skipped": True, "reason": "No GPU"}
            logger.info("  [PHASE 2] Skipped (no GPU available)")

        # --- Phase 3: Hybrid CPU + GPU ---
        if has_gpu:
            logger.info(f"\n  [PHASE 3] Hybrid CPU+GPU ({NUM_PARTITIONS} partitions, split)...")
            phase3 = run_distributed_phase(model_name, input_data, phase="hybrid", use_gpu=True)
            model_results["phase3_hybrid"] = phase3
            if "error" not in phase3:
                logger.info(
                    f"  [PHASE 3] {phase3['throughput_samples_per_sec']:.1f} samples/s | "
                    f"hash={phase3['predictions_hash']}"
                )
            else:
                logger.warning(f"  [PHASE 3] FAILED: {phase3['error'][:80]}")
        else:
            model_results["phase3_hybrid"] = {"phase": "hybrid", "skipped": True, "reason": "No GPU"}
            logger.info("  [PHASE 3] Skipped (no GPU available)")

        all_results[model_name] = model_results

    # --- Generate comparison ---
    comparison = generate_comparison(all_results)
    all_results["_comparison"] = comparison

    # --- Save results ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"cluster_benchmark_{timestamp}.json")

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

    # Print summary
    print_summary(all_results, comparison)

    return all_results


# ---------------------------------------------------------------------------
# Comparison & summary
# ---------------------------------------------------------------------------

def generate_comparison(all_results: Dict) -> Dict[str, Any]:
    """Generate cross-phase comparison metrics."""
    comparison = {}

    for model_name in ALL_MODELS:
        if model_name not in all_results:
            continue

        mr = all_results[model_name]
        model_comp = {"reproducibility": {}, "speedup": {}, "efficiency": {}}

        # Reproducibility
        hashes = {}
        for phase_key, phase_data in mr.items():
            if isinstance(phase_data, dict) and "predictions_hash" in phase_data:
                hashes[phase_key] = phase_data["predictions_hash"]

        model_comp["reproducibility"] = {
            "hashes": hashes,
            "all_match": len(set(hashes.values())) <= 1 if hashes else False,
            "num_phases_compared": len(hashes),
        }

        # Speedup vs baseline
        baseline_tp = mr.get("baseline_cpu", {}).get("throughput_samples_per_sec", 1)
        for phase_key, phase_data in mr.items():
            if phase_key == "baseline_cpu":
                continue
            if isinstance(phase_data, dict) and "throughput_samples_per_sec" in phase_data:
                tp = phase_data["throughput_samples_per_sec"]
                model_comp["speedup"][phase_key] = {
                    "throughput": tp,
                    "vs_baseline": tp / baseline_tp,
                    "time_saved_sec": mr["baseline_cpu"]["total_time_sec"] - phase_data["total_time_sec"],
                }

        # Resource efficiency (throughput per GB memory)
        for phase_key, phase_data in mr.items():
            if isinstance(phase_data, dict) and "executor_metrics" in phase_data:
                em = phase_data["executor_metrics"]
                model_comp["efficiency"][phase_key] = {
                    "partitions_used": phase_data.get("num_partitions", 0),
                    "devices": em.get("devices_used", []),
                    "utilization_balance": (
                        em["min_exec_time_sec"] / em["max_exec_time_sec"]
                        if em["max_exec_time_sec"] > 0 else 0
                    ),
                }

        comparison[model_name] = model_comp

    return comparison


def print_summary(all_results: Dict, comparison: Dict):
    """Print a formatted summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("CLUSTER BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 80)

    # Header
    logger.info(f"\n{'Model':<16} {'Baseline':<12} {'Dist CPU':<12} {'Dist GPU':<12} {'Hybrid':<12} {'Repro'}")
    logger.info("-" * 80)

    for model_name in ALL_MODELS:
        if model_name not in all_results:
            continue

        mr = all_results[model_name]

        baseline = mr.get("baseline_cpu", {}).get("throughput_samples_per_sec", 0)
        p1 = mr.get("phase1_dist_cpu", {})
        p2 = mr.get("phase2_dist_gpu", {})
        p3 = mr.get("phase3_hybrid", {})

        p1_tp = p1.get("throughput_samples_per_sec", 0) if "error" not in p1 else 0
        p2_tp = p2.get("throughput_samples_per_sec", 0) if "error" not in p2 and not p2.get("skipped") else 0
        p3_tp = p3.get("throughput_samples_per_sec", 0) if "error" not in p3 and not p3.get("skipped") else 0

        repro = comparison.get(model_name, {}).get("reproducibility", {}).get("all_match", False)
        repro_str = "✓ PASS" if repro else "✗ FAIL"

        logger.info(
            f"{model_name:<16} "
            f"{baseline:<12.1f} "
            f"{p1_tp:<12.1f} "
            f"{p2_tp:<12.1f} "
            f"{p3_tp:<12.1f} "
            f"{repro_str}"
        )

    # Speedup table
    logger.info(f"\n{'Model':<16} {'CPU Speedup':<14} {'GPU Speedup':<14} {'Hybrid Speedup':<14}")
    logger.info("-" * 60)

    for model_name in ALL_MODELS:
        comp = comparison.get(model_name, {}).get("speedup", {})
        cpu_sp = comp.get("phase1_dist_cpu", {}).get("vs_baseline", 0)
        gpu_sp = comp.get("phase2_dist_gpu", {}).get("vs_baseline", 0)
        hyb_sp = comp.get("phase3_hybrid", {}).get("vs_baseline", 0)

        logger.info(
            f"{model_name:<16} "
            f"{cpu_sp:<14.2f}x "
            f"{gpu_sp:<14.2f}x "
            f"{hyb_sp:<14.2f}x"
        )

    logger.info("\n" + "=" * 80)
    logger.info("(throughput in samples/sec, speedup relative to local CPU baseline)")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_phases()
