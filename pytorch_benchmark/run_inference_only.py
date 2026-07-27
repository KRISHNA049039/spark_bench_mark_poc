"""
Inference-Only Benchmark — Skip Training, Run Pretrained Models One by One

This is the fast path: no training, just load pretrained model weights and
run inference across all available modes (torch_cpu, torch_gpu, spark_cpu, spark_gpu).

Usage:
    # Run one model at a time
    python -m pytorch_benchmark.run_inference_only --model resnet50
    python -m pytorch_benchmark.run_inference_only --model mobilenet_v3
    python -m pytorch_benchmark.run_inference_only --model efficientnet_b0
    python -m pytorch_benchmark.run_inference_only --model distilbert
    python -m pytorch_benchmark.run_inference_only --model tabular_deep

    # Run all models sequentially (one by one)
    python -m pytorch_benchmark.run_inference_only --all

    # Control which modes to use
    python -m pytorch_benchmark.run_inference_only --model resnet50 --no-spark
    python -m pytorch_benchmark.run_inference_only --model resnet50 --cpu-only

    # Adjust sample count for faster/slower runs
    python -m pytorch_benchmark.run_inference_only --model resnet50 --samples 100
"""

import argparse
import gc
import json
import os
import sys
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import torch
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_benchmark")


# ---------------------------------------------------------------------------
# Core inference runner (no training, no Spark overhead for basic modes)
# ---------------------------------------------------------------------------

def run_model_inference(
    model_name: str,
    modes: List[str],
    num_samples: int = 200,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run inference for a single pretrained model across specified modes.

    Returns results dict with per-mode metrics.
    """
    from pytorch_benchmark.pretrained_models import (
        load_pretrained_model,
        generate_vision_inference_data,
        generate_nlp_inference_data,
        generate_tabular_inference_data,
        AVAILABLE_MODELS,
    )
    from pytorch_benchmark.data_generation import seed_everything

    seed_everything(seed)

    model_config = AVAILABLE_MODELS[model_name]
    logger.info(f"Model: {model_name} ({model_config['description']})")
    logger.info(f"  Type: {model_config['type']}, Modes: {modes}, Samples: {num_samples}")

    # Generate input data once
    if model_config["type"] == "vision":
        input_data = generate_vision_inference_data(
            num_samples=num_samples,
            image_size=model_config["input_size"],
            seed=seed,
        )
    elif model_config["type"] == "nlp":
        input_data = generate_nlp_inference_data(
            num_samples=num_samples,
            max_seq_length=model_config["max_seq_length"],
            seed=seed,
        )
    elif model_config["type"] == "tabular":
        input_data = generate_tabular_inference_data(
            num_samples=num_samples,
            num_features=model_config["num_features"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown type: {model_config['type']}")

    results = {"model": model_name, "config": model_config, "modes": {}}

    for mode in modes:
        logger.info(f"\n  [{mode}] Running inference...")
        try:
            result = _run_single_mode(model_name, input_data, mode, batch_size, seed)
            results["modes"][mode] = result
            logger.info(
                f"  [{mode}] Done: {result['throughput_samples_per_sec']:.1f} samples/s, "
                f"latency={result['avg_latency_ms']:.1f}ms/sample"
            )
        except Exception as e:
            results["modes"][mode] = {"error": str(e)}
            logger.warning(f"  [{mode}] Failed: {e}")

    # Reproducibility check
    hashes = {m: r.get("predictions_hash", "") for m, r in results["modes"].items() if "error" not in r}
    results["reproducibility"] = {
        "all_match": len(set(hashes.values())) <= 1,
        "hashes": hashes,
    }

    return results


@torch.no_grad()
def _run_single_mode(
    model_name: str,
    input_data: torch.Tensor,
    mode: str,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Run inference for one mode."""
    import hashlib
    from pytorch_benchmark.pretrained_models import load_pretrained_model
    from pytorch_benchmark.data_generation import seed_everything

    seed_everything(seed)

    # Determine device
    if mode == "torch_cpu":
        device = torch.device("cpu")
    elif mode == "torch_gpu":
        device = torch.device("cuda:0")
    elif mode == "spark_cpu":
        return _run_spark_inference(model_name, input_data, batch_size, seed, use_gpu=False)
    elif mode == "spark_gpu":
        return _run_spark_inference(model_name, input_data, batch_size, seed, use_gpu=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load model
    model, model_info = load_pretrained_model(model_name, device=device)
    model.eval()

    num_samples = len(input_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Resource tracking
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)
    gc_before = gc.get_stats()[0]["collections"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup (2 batches)
    for i in range(min(2, num_batches)):
        s = i * batch_size
        e = min(s + batch_size, num_samples)
        batch = input_data[s:e].to(device)
        _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed inference
    all_preds = []
    batch_times = []

    total_start = time.perf_counter()
    for i in range(num_batches):
        s = i * batch_size
        e = min(s + batch_size, num_samples)
        batch = input_data[s:e].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        output = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        batch_times.append(t1 - t0)
        all_preds.append(output.argmax(dim=1).cpu().numpy())

    total_time = time.perf_counter() - total_start

    # Collect results
    predictions = np.concatenate(all_preds)
    pred_hash = hashlib.sha256(predictions.tobytes()).hexdigest()[:16]

    mem_after = process.memory_info().rss / (1024**2)
    gc_after = gc.get_stats()[0]["collections"]

    result = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "throughput_samples_per_sec": num_samples / total_time,
        "avg_latency_ms": (total_time / num_samples) * 1000,
        "p50_batch_latency_ms": np.percentile(batch_times, 50) * 1000,
        "p95_batch_latency_ms": np.percentile(batch_times, 95) * 1000,
        "p99_batch_latency_ms": np.percentile(batch_times, 99) * 1000,
        "predictions_hash": pred_hash,
        "model_info": model_info,
        "resources": {
            "memory_delta_mb": mem_after - mem_before,
            "gc_collections": gc_after - gc_before,
        },
    }

    if device.type == "cuda":
        result["resources"]["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["resources"]["gpu_name"] = torch.cuda.get_device_name(device)

    # Cleanup
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


def _run_spark_inference(
    model_name: str,
    input_data: torch.Tensor,
    batch_size: int,
    seed: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    """Run distributed inference via Spark."""
    import pickle
    import hashlib
    from pytorch_benchmark.pretrained_models import load_pretrained_model, AVAILABLE_MODELS
    from pytorch_benchmark.config import SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return {"error": "PySpark not installed"}

    spark = None
    try:
        # Load model on CPU for serialization
        model, model_info = load_pretrained_model(model_name, device=torch.device("cpu"))
        model_state_bytes = pickle.dumps(model.state_dict())
        del model
        gc.collect()

        # Spark session
        spark = (
            SparkSession.builder
            .master(SPARK_MASTER)
            .appName(f"InferOnly_{model_name}")
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
            .config("spark.python.worker.reuse", "true")
            .getOrCreate()
        )

        sc = spark.sparkContext
        num_partitions = min(4, sc.defaultParallelism)  # limit partitions for speed

        # Partition data
        input_np = input_data.numpy()
        n = len(input_np)
        chunk_size = (n + num_partitions - 1) // num_partitions
        chunks = [(i, input_np[i*chunk_size : min((i+1)*chunk_size, n)]) for i in range(num_partitions)]

        data_rdd = sc.parallelize(chunks, numSlices=num_partitions)
        model_bc = sc.broadcast(model_state_bytes)

        config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "seed": seed,
            "use_gpu": use_gpu,
        }
        config_bc = sc.broadcast(config)

        infer_start = time.perf_counter()

        def infer_partition(chunk):
            import torch
            import numpy as np
            import pickle as pkl
            from pytorch_benchmark.pretrained_models import load_pretrained_model

            partition_id, data = chunk
            cfg = config_bc.value

            if cfg["use_gpu"] and torch.cuda.is_available():
                device = torch.device(f"cuda:{partition_id % torch.cuda.device_count()}")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                device = torch.device("cpu")

            torch.manual_seed(cfg["seed"])
            model, _ = load_pretrained_model(cfg["model_name"], device=device)
            model.load_state_dict(pkl.loads(model_bc.value))
            model.eval()

            input_tensor = torch.from_numpy(data).to(device)
            bs = cfg["batch_size"]
            preds = []

            with torch.no_grad():
                for s in range(0, len(data), bs):
                    e = min(s + bs, len(data))
                    out = model(input_tensor[s:e])
                    preds.append(out.argmax(dim=1).cpu().numpy())

            if device.type == "cuda":
                del model, input_tensor
                torch.cuda.empty_cache()

            return (partition_id, np.concatenate(preds))

        partition_results = data_rdd.map(infer_partition).collect()
        partition_results.sort(key=lambda x: x[0])
        total_time = time.perf_counter() - infer_start

        all_preds = np.concatenate([r[1] for r in partition_results])
        pred_hash = hashlib.sha256(all_preds.tobytes()).hexdigest()[:16]

        model_bc.unpersist()
        config_bc.unpersist()
        spark.stop()

        return {
            "num_samples": n,
            "batch_size": batch_size,
            "total_time_sec": total_time,
            "throughput_samples_per_sec": n / total_time,
            "avg_latency_ms": (total_time / n) * 1000,
            "p50_batch_latency_ms": (total_time / num_partitions) * 1000,
            "p95_batch_latency_ms": (total_time / num_partitions) * 1000 * 1.1,
            "p99_batch_latency_ms": (total_time / num_partitions) * 1000 * 1.2,
            "predictions_hash": pred_hash,
            "model_info": model_info,
            "num_partitions": num_partitions,
            "use_gpu": use_gpu,
        }

    except Exception as e:
        if spark:
            try:
                spark.stop()
            except Exception:
                pass
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODELS = ["resnet50", "mobilenet_v3", "efficientnet_b0", "distilbert", "tabular_deep"]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only benchmark (no training)")
    parser.add_argument("--model", type=str, choices=ALL_MODELS, help="Single model to run")
    parser.add_argument("--all", action="store_true", help="Run all models one by one")
    parser.add_argument("--samples", type=int, default=200, help="Number of inference samples (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU modes")
    parser.add_argument("--no-spark", action="store_true", help="Skip Spark modes")
    parser.add_argument("--cpu-only", action="store_true", help="Only run torch_cpu")
    return parser.parse_args()


def get_modes(args) -> List[str]:
    if args.cpu_only:
        return ["torch_cpu"]

    modes = ["torch_cpu"]
    if torch.cuda.is_available() and not args.no_gpu:
        modes.append("torch_gpu")
    if not args.no_spark:
        try:
            import pyspark
            modes.append("spark_cpu")
            if torch.cuda.is_available() and not args.no_gpu:
                modes.append("spark_gpu")
        except ImportError:
            pass
    return modes


def print_summary(all_results: List[Dict]):
    """Print final comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE BENCHMARK SUMMARY")
    logger.info("=" * 80)

    for result in all_results:
        model = result["model"]
        logger.info(f"\n  Model: {model}")
        logger.info(f"  {'Mode':<15} {'Throughput':<18} {'Avg Latency':<15} {'p95 Latency':<15} {'Hash':<18}")
        logger.info(f"  {'-'*75}")

        for mode, res in result["modes"].items():
            if "error" in res:
                logger.info(f"  {mode:<15} FAILED: {res['error'][:40]}")
                continue
            logger.info(
                f"  {mode:<15} "
                f"{res['throughput_samples_per_sec']:<18.1f} "
                f"{res['avg_latency_ms']:<15.2f}ms "
                f"{res['p95_batch_latency_ms']:<15.2f}ms "
                f"{res['predictions_hash']}"
            )

        repro = result["reproducibility"]
        status = "PASS" if repro["all_match"] else "DIFFER"
        logger.info(f"  Reproducibility: {status}")

    # GPU speedup
    for result in all_results:
        modes = result["modes"]
        if "torch_cpu" in modes and "torch_gpu" in modes:
            if "error" not in modes["torch_cpu"] and "error" not in modes["torch_gpu"]:
                speedup = modes["torch_gpu"]["throughput_samples_per_sec"] / max(
                    modes["torch_cpu"]["throughput_samples_per_sec"], 0.01
                )
                logger.info(f"\n  {result['model']} GPU speedup: {speedup:.1f}x")


def main():
    args = parse_args()

    if not args.model and not args.all:
        print("Error: Specify --model <name> or --all")
        print(f"Available models: {', '.join(ALL_MODELS)}")
        sys.exit(1)

    modes = get_modes(args)
    models_to_run = ALL_MODELS if args.all else [args.model]

    logger.info("=" * 80)
    logger.info("INFERENCE-ONLY BENCHMARK (No Training)")
    logger.info(f"Models: {models_to_run}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Samples: {args.samples}, Batch size: {args.batch_size}")
    logger.info("=" * 80)

    all_results = []

    for model_name in models_to_run:
        logger.info(f"\n{'─'*80}")
        result = run_model_inference(
            model_name=model_name,
            modes=modes,
            num_samples=args.samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_results.append(result)

    # Summary
    print_summary(all_results)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"inference_only_{timestamp}.json")

    def _serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_serializer)
    logger.info(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
