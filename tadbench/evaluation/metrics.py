"""Detection quality metrics: AUROC, AUPRC, TPR@FPR, calibration error, per-group breakdown."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy import ndarray
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_auroc(y_true: ndarray, y_score: ndarray) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_score: Continuous anomaly scores (higher = more likely malicious).

    Returns:
        AUROC in [0, 1]. Returns 0.0 if only one class is present (undefined).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        logger.warning("compute_auroc: no valid (non-NaN) samples; returning 0.0")
        return 0.0

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "compute_auroc: only one class present (%s); AUROC undefined, returning 0.0",
            unique_classes,
        )
        return 0.0

    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: ndarray, y_score: ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (average precision).

    Args:
        y_true: Binary ground-truth labels (0/1 or bool).
        y_score: Continuous anomaly scores (higher = more likely malicious).

    Returns:
        AUPRC in [0, 1]. Returns 0.0 if only one class is present.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        logger.warning("compute_auprc: no valid (non-NaN) samples; returning 0.0")
        return 0.0

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "compute_auprc: only one class present (%s); AUPRC undefined, returning 0.0",
            unique_classes,
        )
        return 0.0

    return float(average_precision_score(y_true, y_score))


def compute_tpr_at_fpr(
    y_true: ndarray,
    y_score: ndarray,
    target_fprs: list[float] | None = None,
) -> dict[float, float]:
    """Compute True Positive Rate at specific False Positive Rate targets.

    Interpolates linearly on the ROC curve to find TPR at each requested FPR.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Continuous anomaly scores.
        target_fprs: List of FPR values to evaluate at. Defaults to [0.001, 0.005, 0.01].

    Returns:
        Dict mapping each target FPR to the interpolated TPR. Missing/undefined
        entries are 0.0.
    """
    if target_fprs is None:
        target_fprs = [0.001, 0.005, 0.01]

    result: dict[float, float] = {fpr: 0.0 for fpr in target_fprs}

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return result

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)

    for target in target_fprs:
        # Find the highest TPR achievable at FPR <= target via linear interpolation.
        tpr_val = float(np.interp(target, fpr_arr, tpr_arr))
        result[target] = tpr_val

    return result


def compute_calibration_error(
    y_true: ndarray,
    y_score: ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predicted scores into equal-width bins, then computes the weighted
    mean absolute difference between the mean predicted score and the observed
    positive rate in each bin.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Continuous scores in [0, 1] (probability-like).
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]. Returns 0.0 for empty or single-class inputs.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        return 0.0

    # Normalize scores to [0, 1] if they are outside that range.
    score_min = y_score.min()
    score_max = y_score.max()
    if score_max > score_min:
        y_score_norm = (y_score - score_min) / (score_max - score_min)
    else:
        # All scores identical — perfect calibration trivially.
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score_norm, bin_edges[1:-1])  # 0-indexed bins

    ece = 0.0
    n_samples = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        mean_confidence = float(y_score_norm[mask].mean())
        actual_positive_rate = float(y_true[mask].mean())
        bin_weight = mask.sum() / n_samples
        ece += bin_weight * abs(mean_confidence - actual_positive_rate)

    return float(ece)


def compute_per_group_metrics(
    y_true: ndarray,
    y_score: ndarray,
    groups: list[str],
) -> dict[str, dict]:
    """Compute AUROC, AUPRC, and TPR@FPR for each unique group label.

    Useful for per-attack-type and per-length-bin breakdowns.

    Args:
        y_true: Binary ground-truth labels for all samples.
        y_score: Continuous anomaly scores for all samples.
        groups: Group label for each sample (e.g. attack_type strings).

    Returns:
        Dict mapping group name -> dict with keys:
            auroc, auprc, tpr_at_fpr (dict), n_samples, n_positive.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    groups_arr = np.asarray(groups)

    if len(y_true) != len(y_score) or len(y_true) != len(groups_arr):
        raise ValueError(
            "y_true, y_score, and groups must all have the same length; "
            f"got {len(y_true)}, {len(y_score)}, {len(groups_arr)}"
        )

    unique_groups = np.unique(groups_arr)
    result: dict[str, dict] = {}

    for group in unique_groups:
        mask = groups_arr == group
        gt = y_true[mask]
        sc = y_score[mask]

        result[str(group)] = {
            "n_samples": int(mask.sum()),
            "n_positive": int(gt.sum()),
            "auroc": compute_auroc(gt, sc),
            "auprc": compute_auprc(gt, sc),
            "tpr_at_fpr": compute_tpr_at_fpr(gt, sc),
        }

    return result


def compute_all_metrics(
    y_true: ndarray,
    y_score: ndarray,
    groups: Optional[list[str]] = None,
    target_fprs: list[float] | None = None,
    n_calibration_bins: int = 10,
) -> dict:
    """Compute all detection quality metrics in a single call.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Continuous anomaly scores.
        groups: Optional group labels for per-group breakdown (e.g. attack_type).
        target_fprs: FPR targets for TPR@FPR. Defaults to [0.001, 0.005, 0.01].
        n_calibration_bins: Number of bins for calibration error.

    Returns:
        Dict with keys:
            auroc, auprc, tpr_at_fpr, calibration_error,
            n_samples, n_positive, per_group (if groups provided).
    """
    if target_fprs is None:
        target_fprs = [0.001, 0.005, 0.01]

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score_arr)
    n_total = int(len(y_true_arr))
    n_valid = int(valid_mask.sum())
    n_positive = int(y_true_arr[valid_mask].sum())

    metrics: dict = {
        "n_samples": n_total,
        "n_valid": n_valid,
        "n_positive": n_positive,
        "auroc": compute_auroc(y_true_arr, y_score_arr),
        "auprc": compute_auprc(y_true_arr, y_score_arr),
        "tpr_at_fpr": compute_tpr_at_fpr(y_true_arr, y_score_arr, target_fprs),
        "calibration_error": compute_calibration_error(
            y_true_arr, y_score_arr, n_calibration_bins
        ),
    }

    if groups is not None:
        metrics["per_group"] = compute_per_group_metrics(
            y_true_arr, y_score_arr, groups
        )

    return metrics
