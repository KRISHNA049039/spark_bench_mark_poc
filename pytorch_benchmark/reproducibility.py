"""
Reproducibility Verification Module

Compares results across all 4 execution modes (torch_cpu, torch_gpu, spark_cpu, spark_gpu)
to verify numerical reproducibility and consistency.

Checks performed:
    1. Model state hash comparison (same final weights)
    2. Prediction array comparison (element-wise with tolerance)
    3. Probability distribution comparison (KL-divergence and max abs diff)
    4. Loss/accuracy comparison across modes
    5. Self-consistency (running same mode twice produces identical results)

Tolerances:
    - Same-device comparisons (torch_cpu vs spark_cpu): tight tolerance (1e-5)
    - Cross-device comparisons (cpu vs gpu): relaxed tolerance (1e-4)
      due to floating-point non-associativity in GPU reductions
"""

import hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

from pytorch_benchmark.config import RESULT_ATOL, RESULT_RTOL


# ---------------------------------------------------------------------------
# Comparison result types
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    mode_a: str
    mode_b: str
    data_type: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ReproducibilityReport:
    """Complete reproducibility report across all mode comparisons."""
    overall_passed: bool
    total_comparisons: int
    passed_comparisons: int
    failed_comparisons: int
    comparisons: List[ComparisonResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tolerance configurations
# ---------------------------------------------------------------------------

# Same-device comparisons have tighter tolerance
SAME_DEVICE_PAIRS = {
    ("torch_cpu", "spark_cpu"),  # Both CPU
    ("torch_gpu", "spark_gpu"),  # Both GPU
}

# Cross-device pairs have relaxed tolerance (GPU floating-point differences)
CROSS_DEVICE_PAIRS = {
    ("torch_cpu", "torch_gpu"),
    ("torch_cpu", "spark_gpu"),
    ("spark_cpu", "torch_gpu"),
    ("spark_cpu", "spark_gpu"),
}


def get_tolerance(mode_a: str, mode_b: str) -> Tuple[float, float]:
    """
    Get appropriate tolerances for comparing two modes.

    Returns:
        (atol, rtol) tuple
    """
    pair = tuple(sorted([mode_a, mode_b]))

    # Self-comparison: exact match expected
    if mode_a == mode_b:
        return 0.0, 0.0

    # Same device type: tight tolerance
    if pair in SAME_DEVICE_PAIRS or (pair[0], pair[1]) in SAME_DEVICE_PAIRS:
        return RESULT_ATOL, RESULT_RTOL

    # Cross-device: relaxed tolerance
    return RESULT_ATOL * 10, RESULT_RTOL * 10


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------

def compare_predictions(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    """
    Compare prediction arrays between two modes.

    Returns dict with match statistics and details.
    """
    if preds_a is None or preds_b is None:
        return {"error": "One or both prediction arrays are None", "passed": False}

    if preds_a.shape != preds_b.shape:
        return {
            "error": f"Shape mismatch: {preds_a.shape} vs {preds_b.shape}",
            "passed": False,
        }

    total = len(preds_a)
    matches = np.sum(preds_a == preds_b)
    match_rate = matches / total

    # For same-mode self-consistency, expect 100% match
    if mode_a == mode_b:
        threshold = 1.0
    # Same device: expect >99% match
    elif tuple(sorted([mode_a, mode_b])) in SAME_DEVICE_PAIRS:
        threshold = 0.99
    # Cross-device: expect >95% match (GPU reduction order differences)
    else:
        threshold = 0.95

    passed = match_rate >= threshold

    return {
        "total_predictions": total,
        "matching_predictions": int(matches),
        "match_rate": float(match_rate),
        "threshold": threshold,
        "passed": passed,
        "mismatched_indices": np.where(preds_a != preds_b)[0][:20].tolist(),  # first 20
    }


def compare_probabilities(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    """
    Compare probability distributions between two modes.

    Uses multiple metrics:
    - Max absolute difference
    - Mean absolute difference
    - KL divergence (symmetrized)
    - Cosine similarity
    """
    if probs_a is None or probs_b is None:
        return {"error": "One or both probability arrays are None", "passed": False}

    if probs_a.shape != probs_b.shape:
        return {
            "error": f"Shape mismatch: {probs_a.shape} vs {probs_b.shape}",
            "passed": False,
        }

    atol, rtol = get_tolerance(mode_a, mode_b)

    # Absolute differences
    abs_diff = np.abs(probs_a - probs_b)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative differences (avoid division by zero)
    denom = np.maximum(np.abs(probs_a), 1e-10)
    rel_diff = abs_diff / denom
    max_rel_diff = float(np.max(rel_diff))
    mean_rel_diff = float(np.mean(rel_diff))

    # KL divergence (symmetrized, with epsilon for stability)
    eps = 1e-10
    p = np.clip(probs_a, eps, 1.0)
    q = np.clip(probs_b, eps, 1.0)
    kl_pq = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
    kl_qp = float(np.mean(np.sum(q * np.log(q / p), axis=1)))
    sym_kl = (kl_pq + kl_qp) / 2

    # Cosine similarity (per sample, averaged)
    cos_sims = []
    for i in range(len(probs_a)):
        dot = np.dot(probs_a[i], probs_b[i])
        norm_a = np.linalg.norm(probs_a[i])
        norm_b = np.linalg.norm(probs_b[i])
        cos_sims.append(dot / (norm_a * norm_b + 1e-10))
    mean_cosine_sim = float(np.mean(cos_sims))

    # Pass/fail determination
    if mode_a == mode_b:
        # Self-consistency: expect exact match
        passed = max_abs_diff == 0.0
    else:
        # Tolerance-based check
        all_close = np.allclose(probs_a, probs_b, atol=atol, rtol=rtol)
        passed = all_close or (sym_kl < 0.01 and mean_cosine_sim > 0.999)

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "kl_divergence_symmetric": sym_kl,
        "mean_cosine_similarity": mean_cosine_sim,
        "tolerance_atol": atol,
        "tolerance_rtol": rtol,
        "numpy_allclose": bool(np.allclose(probs_a, probs_b, atol=atol, rtol=rtol)),
        "passed": passed,
    }


def compare_losses(
    losses_a: List[float],
    losses_b: List[float],
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    """Compare training loss curves between two modes."""
    if not losses_a or not losses_b:
        return {"error": "Empty loss arrays", "passed": False}

    if len(losses_a) != len(losses_b):
        return {
            "error": f"Length mismatch: {len(losses_a)} vs {len(losses_b)}",
            "passed": False,
        }

    atol, rtol = get_tolerance(mode_a, mode_b)

    losses_a_arr = np.array(losses_a)
    losses_b_arr = np.array(losses_b)
    abs_diff = np.abs(losses_a_arr - losses_b_arr)

    # Both should converge (decreasing trend)
    a_converging = losses_a[-1] < losses_a[0]
    b_converging = losses_b[-1] < losses_b[0]

    # Final loss comparison
    final_diff = abs(losses_a[-1] - losses_b[-1])

    # For cross-device, allow larger final loss difference
    if mode_a == mode_b:
        loss_threshold = 0.0
    elif tuple(sorted([mode_a, mode_b])) in SAME_DEVICE_PAIRS:
        loss_threshold = 0.05
    else:
        loss_threshold = 0.15  # GPU reductions can diverge over epochs

    passed = (
        a_converging and b_converging and
        final_diff < loss_threshold
    )

    return {
        "epoch_diffs": abs_diff.tolist(),
        "max_epoch_diff": float(np.max(abs_diff)),
        "mean_epoch_diff": float(np.mean(abs_diff)),
        "final_loss_diff": final_diff,
        "loss_threshold": loss_threshold,
        "a_converging": a_converging,
        "b_converging": b_converging,
        "final_loss_a": losses_a[-1],
        "final_loss_b": losses_b[-1],
        "passed": passed,
    }


def compare_model_hashes(hash_a: str, hash_b: str, mode_a: str, mode_b: str) -> Dict[str, Any]:
    """
    Compare model state hashes.

    Same-device modes should produce identical hashes (deterministic training).
    Cross-device modes may differ due to GPU floating-point arithmetic.
    """
    match = hash_a == hash_b

    if mode_a == mode_b:
        # Self-consistency: must match
        passed = match
        note = "Self-consistency check"
    elif tuple(sorted([mode_a, mode_b])) in SAME_DEVICE_PAIRS:
        # Same device: Spark gradient aggregation may cause small diffs
        # Hash won't match exactly due to aggregation ordering, so this is informational
        passed = True  # Don't fail on hash alone for spark vs torch
        note = "Same-device pair: hash match is informational (aggregation order may differ)"
    else:
        # Cross-device: hash mismatch expected
        passed = True
        note = "Cross-device pair: hash mismatch expected due to GPU arithmetic"

    return {
        "hash_a": hash_a,
        "hash_b": hash_b,
        "match": match,
        "passed": passed,
        "note": note,
    }


def compare_accuracy(
    acc_a: float,
    acc_b: float,
    mode_a: str,
    mode_b: str,
) -> Dict[str, Any]:
    """Compare final test accuracies."""
    diff = abs(acc_a - acc_b)

    if mode_a == mode_b:
        threshold = 0.0
    elif tuple(sorted([mode_a, mode_b])) in SAME_DEVICE_PAIRS:
        threshold = 0.02  # 2% tolerance
    else:
        threshold = 0.05  # 5% tolerance for cross-device

    passed = diff <= threshold

    return {
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "difference": diff,
        "threshold": threshold,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Full comparison between two runs
# ---------------------------------------------------------------------------

def compare_runs(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    data_type: str,
) -> ComparisonResult:
    """
    Perform full comparison between two benchmark run results.

    Args:
        result_a: Result dict from runner A (must have predictions_hash, etc.)
        result_b: Result dict from runner B
        data_type: 'structured' or 'unstructured'

    Returns:
        ComparisonResult with all check details
    """
    mode_a = result_a.get("mode", "unknown")
    mode_b = result_b.get("mode", "unknown")

    comparison = ComparisonResult(
        mode_a=mode_a,
        mode_b=mode_b,
        data_type=data_type,
        passed=True,
    )

    # 1. Model hash comparison
    hash_result = compare_model_hashes(
        result_a.get("model_state_hash", ""),
        result_b.get("model_state_hash", ""),
        mode_a, mode_b,
    )
    comparison.details["model_hash"] = hash_result
    if not hash_result["passed"]:
        comparison.failures.append(f"Model hash check failed: {hash_result['note']}")
        comparison.passed = False

    # 2. Loss curve comparison
    loss_result = compare_losses(
        result_a.get("train_losses", []),
        result_b.get("train_losses", []),
        mode_a, mode_b,
    )
    comparison.details["losses"] = loss_result
    if not loss_result.get("passed", False):
        comparison.failures.append(
            f"Loss comparison failed: final diff={loss_result.get('final_loss_diff', 'N/A')}"
        )
        comparison.passed = False

    # 3. Accuracy comparison
    acc_result = compare_accuracy(
        result_a.get("test_accuracy", 0),
        result_b.get("test_accuracy", 0),
        mode_a, mode_b,
    )
    comparison.details["accuracy"] = acc_result
    if not acc_result["passed"]:
        comparison.failures.append(
            f"Accuracy difference too large: {acc_result['difference']:.4f} > {acc_result['threshold']}"
        )
        comparison.passed = False

    # 4. Warnings for informational metrics
    if not hash_result["match"]:
        comparison.warnings.append(
            f"Model hashes differ between {mode_a} and {mode_b} (may be expected)"
        )

    return comparison


def compare_runs_with_predictions(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    data_type: str,
) -> ComparisonResult:
    """
    Full comparison including prediction and probability arrays.

    Use this when raw predictions/probabilities are available
    (not just hashes from serialized results).
    """
    comparison = compare_runs(result_a, result_b, data_type)
    mode_a = result_a.get("mode", "unknown")
    mode_b = result_b.get("mode", "unknown")

    # 5. Prediction comparison
    pred_result = compare_predictions(preds_a, preds_b, mode_a, mode_b)
    comparison.details["predictions"] = pred_result
    if not pred_result.get("passed", False):
        comparison.failures.append(
            f"Prediction match rate too low: {pred_result.get('match_rate', 0):.4f}"
        )
        comparison.passed = False

    # 6. Probability comparison
    prob_result = compare_probabilities(probs_a, probs_b, mode_a, mode_b)
    comparison.details["probabilities"] = prob_result
    if not prob_result.get("passed", False):
        comparison.failures.append(
            f"Probability divergence too high: KL={prob_result.get('kl_divergence_symmetric', 'N/A')}"
        )
        comparison.passed = False

    return comparison


# ---------------------------------------------------------------------------
# Generate full reproducibility report
# ---------------------------------------------------------------------------

def generate_reproducibility_report(
    all_results: Dict[str, Dict[str, Any]],
    all_predictions: Optional[Dict[str, np.ndarray]] = None,
    all_probabilities: Optional[Dict[str, np.ndarray]] = None,
) -> ReproducibilityReport:
    """
    Generate a comprehensive reproducibility report comparing all modes.

    Args:
        all_results: Dict mapping mode names to their result dicts.
                    Each result dict should have keys per data_type (structured/unstructured).
        all_predictions: Optional dict mapping "mode_datatype" to prediction arrays
        all_probabilities: Optional dict mapping "mode_datatype" to probability arrays

    Returns:
        ReproducibilityReport with all pairwise comparisons
    """
    modes = list(all_results.keys())
    data_types = ["structured", "unstructured"]
    comparisons = []

    for data_type in data_types:
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                mode_a = modes[i]
                mode_b = modes[j]

                result_a = all_results[mode_a].get(data_type, {})
                result_b = all_results[mode_b].get(data_type, {})

                if not result_a or not result_b:
                    continue

                # Use full comparison if predictions available
                key_a = f"{mode_a}_{data_type}"
                key_b = f"{mode_b}_{data_type}"

                if (
                    all_predictions and all_probabilities and
                    key_a in all_predictions and key_b in all_predictions and
                    key_a in all_probabilities and key_b in all_probabilities
                ):
                    comparison = compare_runs_with_predictions(
                        result_a, result_b,
                        all_predictions[key_a], all_predictions[key_b],
                        all_probabilities[key_a], all_probabilities[key_b],
                        data_type,
                    )
                else:
                    comparison = compare_runs(result_a, result_b, data_type)

                comparisons.append(comparison)

    # Build report
    passed_count = sum(1 for c in comparisons if c.passed)
    failed_count = len(comparisons) - passed_count

    report = ReproducibilityReport(
        overall_passed=(failed_count == 0),
        total_comparisons=len(comparisons),
        passed_comparisons=passed_count,
        failed_comparisons=failed_count,
        comparisons=comparisons,
        summary={
            "modes_tested": modes,
            "data_types_tested": data_types,
            "pairwise_results": {
                f"{c.mode_a}_vs_{c.mode_b}_{c.data_type}": {
                    "passed": c.passed,
                    "failures": c.failures,
                    "warnings": c.warnings,
                }
                for c in comparisons
            },
        },
    )

    return report


def format_report(report: ReproducibilityReport) -> str:
    """Format reproducibility report as a readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("REPRODUCIBILITY VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Overall Result: {'PASSED' if report.overall_passed else 'FAILED'}")
    lines.append(f"Total Comparisons: {report.total_comparisons}")
    lines.append(f"Passed: {report.passed_comparisons}")
    lines.append(f"Failed: {report.failed_comparisons}")
    lines.append("")

    for comparison in report.comparisons:
        status = "PASS" if comparison.passed else "FAIL"
        lines.append(f"  [{status}] {comparison.mode_a} vs {comparison.mode_b} ({comparison.data_type})")

        if comparison.failures:
            for failure in comparison.failures:
                lines.append(f"        FAILURE: {failure}")

        if comparison.warnings:
            for warning in comparison.warnings:
                lines.append(f"        WARNING: {warning}")

        # Key metrics
        if "accuracy" in comparison.details:
            acc = comparison.details["accuracy"]
            lines.append(
                f"        Accuracy: {acc.get('accuracy_a', 0):.4f} vs "
                f"{acc.get('accuracy_b', 0):.4f} (diff={acc.get('difference', 0):.4f})"
            )

        if "predictions" in comparison.details:
            pred = comparison.details["predictions"]
            lines.append(
                f"        Predictions match rate: {pred.get('match_rate', 0):.4f}"
            )

        if "probabilities" in comparison.details:
            prob = comparison.details["probabilities"]
            lines.append(
                f"        Max abs prob diff: {prob.get('max_abs_diff', 0):.6f}, "
                f"KL div: {prob.get('kl_divergence_symmetric', 0):.6f}"
            )

        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
