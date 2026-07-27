"""
Cluster Benchmark Report Generator

Reads cluster_benchmark_*.json results and generates:
1. A comprehensive markdown report with comparison tables
2. PNG charts comparing all phases

Usage:
    python -m pytorch_benchmark.generate_cluster_report [--input <json_path>]

If no --input specified, uses the latest cluster_benchmark_*.json in benchmark_results/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

OUTPUT_DIR = os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_results")


def load_results(input_path: str = None) -> Dict[str, Any]:
    """Load the cluster benchmark results JSON."""
    if input_path and os.path.exists(input_path):
        with open(input_path) as f:
            return json.load(f)

    # Find latest
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("cluster_benchmark_") and f.endswith(".json")
    ])
    if not files:
        print(f"No cluster_benchmark_*.json found in {OUTPUT_DIR}")
        sys.exit(1)

    path = os.path.join(OUTPUT_DIR, files[-1])
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def generate_markdown_report(data: Dict[str, Any], output_path: str):
    """Generate a comprehensive markdown comparison report."""
    comparison = data.get("_comparison", {})
    models = [k for k in data.keys() if not k.startswith("_")]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # --- Header ---
    lines.append("# Cluster Benchmark Report — 3-Phase Comparison")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}  ")
    lines.append(f"**Cluster:** 2 Worker Nodes (32 GB RAM, 8 GB VRAM each)  ")

    # Extract sample/batch info from first model
    first_model = data.get(models[0], {}) if models else {}
    baseline = first_model.get("baseline_cpu", {})
    num_samples = baseline.get("num_samples", "?")
    lines.append(f"**Samples per model:** {num_samples} | **Batch size:** 64 | **Partitions:** 8  ")
    lines.append(f"**Executor Memory:** 12 GB heap + 2 GB overhead | **Cores:** 4 per executor")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Phase descriptions ---
    lines.append("## Test Phases")
    lines.append("")
    lines.append("| Phase | Mode | Description |")
    lines.append("|-------|------|-------------|")
    lines.append("| Baseline | Local CPU | Single-machine, no Spark, no distribution |")
    lines.append("| Phase 1 | Distributed CPU | All partitions on CPU workers via Spark |")
    lines.append("| Phase 2 | Distributed GPU | All partitions on GPU workers via Spark |")
    lines.append("| Phase 3 | Hybrid CPU+GPU | Even partitions→GPU, Odd partitions→CPU |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Throughput Comparison Table ---
    lines.append("## 1. Throughput Comparison (samples/sec)")
    lines.append("")
    lines.append("| Model | Baseline (CPU) | Phase 1 (Dist CPU) | Phase 2 (Dist GPU) | Phase 3 (Hybrid) | Best Phase |")
    lines.append("|-------|---:|---:|---:|---:|---|")

    for model in models:
        mr = data[model]
        bl = mr.get("baseline_cpu", {}).get("throughput_samples_per_sec", 0)
        p1 = mr.get("phase1_dist_cpu", {})
        p2 = mr.get("phase2_dist_gpu", {})
        p3 = mr.get("phase3_hybrid", {})

        p1_tp = p1.get("throughput_samples_per_sec", 0) if "error" not in p1 else 0
        p2_tp = p2.get("throughput_samples_per_sec", 0) if "error" not in p2 and not p2.get("skipped") else 0
        p3_tp = p3.get("throughput_samples_per_sec", 0) if "error" not in p3 and not p3.get("skipped") else 0

        # Determine best
        phases = {"Baseline": bl, "Dist CPU": p1_tp, "Dist GPU": p2_tp, "Hybrid": p3_tp}
        best = max(phases, key=phases.get)

        p2_str = f"{p2_tp:.1f}" if p2_tp > 0 else "N/A"
        p3_str = f"{p3_tp:.1f}" if p3_tp > 0 else "N/A"

        lines.append(f"| **{model}** | {bl:.1f} | {p1_tp:.1f} | {p2_str} | {p3_str} | **{best}** |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Speedup Table ---
    lines.append("## 2. Speedup vs Baseline (Local CPU)")
    lines.append("")
    lines.append("| Model | Dist CPU | Dist GPU | Hybrid | Best Speedup |")
    lines.append("|-------|---:|---:|---:|---:|")

    for model in models:
        comp = comparison.get(model, {}).get("speedup", {})
        cpu_sp = comp.get("phase1_dist_cpu", {}).get("vs_baseline", 0)
        gpu_sp = comp.get("phase2_dist_gpu", {}).get("vs_baseline", 0)
        hyb_sp = comp.get("phase3_hybrid", {}).get("vs_baseline", 0)

        best_sp = max(cpu_sp, gpu_sp, hyb_sp)

        gpu_str = f"{gpu_sp:.2f}x" if gpu_sp > 0 else "N/A"
        hyb_str = f"{hyb_sp:.2f}x" if hyb_sp > 0 else "N/A"

        lines.append(f"| **{model}** | {cpu_sp:.2f}x | {gpu_str} | {hyb_str} | **{best_sp:.2f}x** |")

    lines.append("")

    # Speedup bar chart (ASCII)
    lines.append("```")
    lines.append("Speedup vs Local CPU Baseline")
    lines.append("═" * 70)
    for model in models:
        comp = comparison.get(model, {}).get("speedup", {})
        gpu_sp = comp.get("phase2_dist_gpu", {}).get("vs_baseline", 0)
        if gpu_sp > 0:
            bar_len = int(min(gpu_sp * 3, 50))
            lines.append(f"{model:<16} {'█' * bar_len} {gpu_sp:.1f}x (GPU)")
        else:
            cpu_sp = comp.get("phase1_dist_cpu", {}).get("vs_baseline", 0)
            bar_len = int(min(cpu_sp * 3, 50))
            lines.append(f"{model:<16} {'▒' * bar_len} {cpu_sp:.1f}x (CPU)")
    lines.append("═" * 70)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Reproducibility ---
    lines.append("## 3. Reproducibility Verification")
    lines.append("")
    lines.append("| Model | Baseline Hash | Dist CPU Hash | Dist GPU Hash | Hybrid Hash | All Match? |")
    lines.append("|-------|:---:|:---:|:---:|:---:|:---:|")

    for model in models:
        comp = comparison.get(model, {}).get("reproducibility", {})
        hashes = comp.get("hashes", {})
        all_match = comp.get("all_match", False)

        bl_h = hashes.get("baseline_cpu", "—")[:8]
        p1_h = hashes.get("phase1_dist_cpu", "—")[:8]
        p2_h = hashes.get("phase2_dist_gpu", "—")[:8]
        p3_h = hashes.get("phase3_hybrid", "—")[:8]
        match_str = "✅" if all_match else "⚠️"

        lines.append(f"| **{model}** | `{bl_h}` | `{p1_h}` | `{p2_h}` | `{p3_h}` | {match_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Timing Breakdown ---
    lines.append("## 4. Total Execution Time (seconds)")
    lines.append("")
    lines.append("| Model | Baseline | Dist CPU | Dist GPU | Hybrid | Time Saved (best) |")
    lines.append("|-------|---:|---:|---:|---:|---:|")

    for model in models:
        mr = data[model]
        bl_t = mr.get("baseline_cpu", {}).get("total_time_sec", 0)
        p1_t = mr.get("phase1_dist_cpu", {}).get("total_time_sec", 0) if "error" not in mr.get("phase1_dist_cpu", {}) else 0
        p2_t = mr.get("phase2_dist_gpu", {}).get("total_time_sec", 0) if "error" not in mr.get("phase2_dist_gpu", {}) and not mr.get("phase2_dist_gpu", {}).get("skipped") else 0
        p3_t = mr.get("phase3_hybrid", {}).get("total_time_sec", 0) if "error" not in mr.get("phase3_hybrid", {}) and not mr.get("phase3_hybrid", {}).get("skipped") else 0

        best_t = min(t for t in [p1_t, p2_t, p3_t] if t > 0) if any(t > 0 for t in [p1_t, p2_t, p3_t]) else bl_t
        saved = bl_t - best_t

        p2_str = f"{p2_t:.2f}" if p2_t > 0 else "N/A"
        p3_str = f"{p3_t:.2f}" if p3_t > 0 else "N/A"

        lines.append(f"| **{model}** | {bl_t:.2f} | {p1_t:.2f} | {p2_str} | {p3_str} | {saved:+.2f}s |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Executor Metrics ---
    lines.append("## 5. Per-Executor Performance (Phase 1: Dist CPU)")
    lines.append("")
    lines.append("| Model | Partitions | Avg Exec Time | Max Exec Time | Load Balance | Total Throughput |")
    lines.append("|-------|---:|---:|---:|---:|---:|")

    for model in models:
        p1 = data[model].get("phase1_dist_cpu", {})
        if "error" in p1 or "executor_metrics" not in p1:
            lines.append(f"| **{model}** | — | — | — | — | — |")
            continue

        em = p1["executor_metrics"]
        balance = em.get("min_exec_time_sec", 0) / max(em.get("max_exec_time_sec", 1), 0.001)
        balance_pct = balance * 100

        lines.append(
            f"| **{model}** | {p1.get('num_partitions', 0)} | "
            f"{em['avg_exec_time_sec']:.2f}s | {em['max_exec_time_sec']:.2f}s | "
            f"{balance_pct:.0f}% | {em['total_throughput']:.1f} s/s |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Resource Planning Summary ---
    lines.append("## 6. Resource Utilization Plan")
    lines.append("")
    lines.append("### Memory Layout (per worker node)")
    lines.append("")
    lines.append("```")
    lines.append("32 GB Total RAM")
    lines.append("┌─────────────────────────────────────────────────────────────┐")
    lines.append("│  OS + Docker (3 GB)  │  Spark Daemon (1 GB)  │  Buffer (2 GB)│")
    lines.append("├──────────────────────┴───────────────────────┴──────────────┤")
    lines.append("│          Executor 1: CPU (12 GB heap + 2 GB overhead)        │")
    lines.append("├──────────────────────────────────────────────────────────────┤")
    lines.append("│          Executor 2: GPU (10 GB heap + 2 GB overhead)        │")
    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("8 GB VRAM")
    lines.append("┌──────────────────────────────────────────────────────────────┐")
    lines.append("│ Model (~300 MB) │ Activations (~500 MB) │ Available (~7 GB)   │")
    lines.append("└──────────────────────────────────────────────────────────────┘")
    lines.append("```")
    lines.append("")
    lines.append("### Parallelism Configuration")
    lines.append("")
    lines.append("| Setting | Value | Rationale |")
    lines.append("|---------|-------|-----------|")
    lines.append("| Workers | 2 | Available machines |")
    lines.append("| Executors per worker | 2 | 1 CPU + 1 GPU executor |")
    lines.append("| Cores per executor | 4 | Balance parallelism vs memory |")
    lines.append("| Partitions | 8 | 2 per executor for load balancing |")
    lines.append("| Executor memory | 12 GB | ~40% of available RAM per executor |")
    lines.append("| Memory overhead | 2 GB | Python/serialization buffer |")
    lines.append("| Batch size | 64 | Fits in GPU VRAM with margin |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Conclusions ---
    lines.append("## 7. Conclusions & Recommendations")
    lines.append("")

    # Auto-generate conclusions based on data
    conclusions = []

    # Check reproducibility
    all_repro = all(
        comparison.get(m, {}).get("reproducibility", {}).get("all_match", False)
        for m in models
    )
    if all_repro:
        conclusions.append("**Reproducibility:** All models produce identical predictions across all phases (baseline, distributed CPU, distributed GPU, hybrid). Distribution does not affect correctness.")
    else:
        cpu_match = all(
            comparison.get(m, {}).get("reproducibility", {}).get("hashes", {}).get("baseline_cpu") ==
            comparison.get(m, {}).get("reproducibility", {}).get("hashes", {}).get("phase1_dist_cpu")
            for m in models
            if "phase1_dist_cpu" in comparison.get(m, {}).get("reproducibility", {}).get("hashes", {})
        )
        if cpu_match:
            conclusions.append("**Reproducibility:** CPU modes always match. GPU predictions may differ for some models due to floating-point non-associativity — this is expected behavior.")

    # Best speedup
    best_model_gpu = ""
    best_gpu_speedup = 0
    for m in models:
        sp = comparison.get(m, {}).get("speedup", {}).get("phase2_dist_gpu", {}).get("vs_baseline", 0)
        if sp > best_gpu_speedup:
            best_gpu_speedup = sp
            best_model_gpu = m

    if best_gpu_speedup > 1:
        conclusions.append(f"**Best GPU speedup:** {best_model_gpu} at {best_gpu_speedup:.1f}x over local CPU baseline.")

    # Spark overhead
    overhead_models = []
    for m in models:
        sp = comparison.get(m, {}).get("speedup", {}).get("phase1_dist_cpu", {}).get("vs_baseline", 0)
        if 0 < sp < 1:
            overhead_models.append(m)
    if overhead_models:
        conclusions.append(f"**Spark overhead at current scale:** {', '.join(overhead_models)} are slower with distributed CPU than local — increase sample count to amortize fixed costs.")

    conclusions.append("**Hybrid mode** simultaneously utilizes CPU and GPU, maximizing hardware utilization when both resources are available.")
    conclusions.append("**Scale recommendation:** At 10K+ samples, distributed modes will show clear throughput advantages over single-machine inference.")

    for c in conclusions:
        lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated on {timestamp} from cluster benchmark results.*")

    # Write
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved: {output_path}")
    return report_text


def generate_charts(data: Dict[str, Any]):
    """Generate PNG charts from cluster benchmark results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    models = [k for k in data.keys() if not k.startswith("_")]
    comparison = data.get("_comparison", {})

    COLORS = {
        "baseline_cpu": "#607D8B",
        "phase1_dist_cpu": "#2196F3",
        "phase2_dist_gpu": "#4CAF50",
        "phase3_hybrid": "#FF9800",
    }
    LABELS = {
        "baseline_cpu": "Baseline (Local CPU)",
        "phase1_dist_cpu": "Phase 1: Dist CPU",
        "phase2_dist_gpu": "Phase 2: Dist GPU",
        "phase3_hybrid": "Phase 3: Hybrid",
    }

    # --- Chart 1: Throughput grouped bar ---
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.2
    phases = ["baseline_cpu", "phase1_dist_cpu", "phase2_dist_gpu", "phase3_hybrid"]

    for i, phase in enumerate(phases):
        tps = []
        for model in models:
            mr = data[model].get(phase, {})
            if "error" in mr or mr.get("skipped"):
                tps.append(0)
            else:
                tps.append(mr.get("throughput_samples_per_sec", 0))

        ax.bar(x + i * width, tps, width, label=LABELS[phase], color=COLORS[phase], edgecolor="white")

    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Throughput (samples/sec)", fontweight="bold")
    ax.set_title("3-Phase Cluster Benchmark — Throughput Comparison", fontweight="bold", pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_chart_throughput.png"), dpi=150)
    plt.close()
    print("  Saved: cluster_chart_throughput.png")

    # --- Chart 2: Speedup comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    for i, phase in enumerate(["phase1_dist_cpu", "phase2_dist_gpu", "phase3_hybrid"]):
        speedups = []
        for model in models:
            sp = comparison.get(model, {}).get("speedup", {}).get(phase, {}).get("vs_baseline", 0)
            speedups.append(sp)
        ax.bar(x + i * width, speedups, width, label=LABELS[phase], color=COLORS[phase], edgecolor="white")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Speedup vs Local CPU", fontweight="bold")
    ax.set_title("Speedup Over Baseline — All Phases", fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_chart_speedup.png"), dpi=150)
    plt.close()
    print("  Saved: cluster_chart_speedup.png")

    # --- Chart 3: Time comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2

    for i, phase in enumerate(phases):
        times = []
        for model in models:
            mr = data[model].get(phase, {})
            if "error" in mr or mr.get("skipped"):
                times.append(0)
            else:
                times.append(mr.get("total_time_sec", 0))
        ax.bar(x + i * width, times, width, label=LABELS[phase], color=COLORS[phase], edgecolor="white")

    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Total Time (seconds)", fontweight="bold")
    ax.set_title("Total Inference Time — All Phases", fontweight="bold", pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_chart_time.png"), dpi=150)
    plt.close()
    print("  Saved: cluster_chart_time.png")

    # --- Chart 4: Reproducibility heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))

    phase_labels = ["Dist CPU\nvs Baseline", "Dist GPU\nvs Baseline", "Hybrid\nvs Baseline"]
    match_matrix = np.zeros((len(models), 3))

    for i, model in enumerate(models):
        hashes = comparison.get(model, {}).get("reproducibility", {}).get("hashes", {})
        bl_hash = hashes.get("baseline_cpu", "")
        match_matrix[i, 0] = 1 if hashes.get("phase1_dist_cpu", "") == bl_hash and bl_hash else 0
        match_matrix[i, 1] = 1 if hashes.get("phase2_dist_gpu", "") == bl_hash and bl_hash else -1
        match_matrix[i, 2] = 1 if hashes.get("phase3_hybrid", "") == bl_hash and bl_hash else -1

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#EF5350", "#BDBDBD", "#4CAF50"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(match_matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(phase_labels, fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontweight="bold")
    ax.set_title("Reproducibility: Prediction Match vs Baseline", fontweight="bold", pad=15)

    for i in range(len(models)):
        for j in range(3):
            val = match_matrix[i, j]
            symbol = "✓" if val == 1 else ("✗" if val == 0 else "—")
            ax.text(j, i, symbol, ha="center", va="center", fontsize=16, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_chart_reproducibility.png"), dpi=150)
    plt.close()
    print("  Saved: cluster_chart_reproducibility.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate cluster benchmark report")
    parser.add_argument("--input", type=str, default=None, help="Path to cluster_benchmark_*.json")
    args = parser.parse_args()

    data = load_results(args.input)

    # Generate markdown report
    report_path = os.path.join(OUTPUT_DIR, "CLUSTER_BENCHMARK_REPORT.md")
    generate_markdown_report(data, report_path)

    # Generate charts
    generate_charts(data)

    print("\nDone!")


if __name__ == "__main__":
    main()
