"""Threshold calibration: find the score threshold that achieves a target FPR."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


def calibrate_threshold(
    y_true: ndarray,
    y_score: ndarray,
    target_fpr: float,
) -> float:
    """Find the score threshold that achieves FPR <= target_fpr.

    Computes the ROC curve and returns the threshold with highest TPR where the
    false positive rate does not exceed ``target_fpr``.  If no such threshold
    exists (e.g. target_fpr is lower than the minimum achievable FPR for the
    given data), returns the threshold corresponding to the minimum FPR.

    Args:
        y_true: Binary ground-truth labels (0 = benign, 1 = malicious).
        y_score: Continuous anomaly scores (higher = more likely malicious).
        target_fpr: Maximum acceptable false positive rate (e.g. 0.005).

    Returns:
        Score threshold (float).  Samples with score >= threshold are classified
        as malicious.

    Raises:
        ValueError: If y_true contains only one class (ROC undefined).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = ~np.isnan(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        raise ValueError("calibrate_threshold: no valid (non-NaN) samples")

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError(
            f"calibrate_threshold: only one class present ({unique_classes}); "
            "ROC curve is undefined."
        )

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    # sklearn roc_curve returns FPR/TPR/thresholds sorted by increasing FPR.
    # thresholds[0] > thresholds[-1]: the first threshold classifies everything
    # as negative (FPR=0, TPR=0), the last classifies everything as positive.

    # Find indices where FPR <= target_fpr.
    feasible = np.where(fpr_arr <= target_fpr)[0]
    if len(feasible) == 0:
        # Target FPR unachievable — fall back to the threshold giving minimum FPR.
        logger.warning(
            "calibrate_threshold: target FPR %.4f is below minimum achievable FPR %.4f; "
            "using threshold corresponding to minimum FPR",
            target_fpr,
            fpr_arr.min(),
        )
        best_idx = int(np.argmin(fpr_arr))
    else:
        # Among feasible operating points, pick the one with the highest TPR
        # (i.e. the one closest to target_fpr from below).
        best_idx = int(feasible[np.argmax(tpr_arr[feasible])])

    threshold = float(thresholds[best_idx])
    logger.debug(
        "calibrate_threshold: target_fpr=%.4f -> threshold=%.6f "
        "(actual FPR=%.4f, TPR=%.4f)",
        target_fpr,
        threshold,
        fpr_arr[best_idx],
        tpr_arr[best_idx],
    )
    return threshold


def calibrate_all_detectors(
    detectors_scores: dict[str, tuple[ndarray, ndarray]],
    target_fpr: float,
) -> dict[str, float]:
    """Calibrate thresholds for multiple detectors at the same target FPR.

    Args:
        detectors_scores: Mapping of detector name -> (y_true, y_score).
        target_fpr: Target false positive rate for all detectors.

    Returns:
        Mapping of detector name -> calibrated threshold.  Detectors for which
        calibration fails (e.g. single-class data) are omitted with a warning.
    """
    thresholds: dict[str, float] = {}
    for name, (y_true, y_score) in detectors_scores.items():
        try:
            thresholds[name] = calibrate_threshold(y_true, y_score, target_fpr)
        except ValueError as exc:
            logger.warning("calibrate_all_detectors: skipping '%s': %s", name, exc)
    return thresholds


def save_thresholds(
    thresholds: dict[str, float],
    path: Union[str, Path],
    target_fpr: float,
    dataset_name: str,
    detector_configs: dict | None = None,
) -> None:
    """Persist calibrated thresholds to a JSON file with metadata.

    The JSON structure is::

        {
            "target_fpr": 0.005,
            "calibration_dataset": "benchmark_val",
            "timestamp": "2026-02-23T12:00:00+00:00",
            "thresholds": {
                "threat_prototype": 0.732,
                "intra_prompt_outlier": 0.441,
                ...
            }
        }

    Args:
        thresholds: Mapping of detector name -> threshold value.
        path: Destination file path (parent directories created as needed).
        target_fpr: The FPR used during calibration (stored as metadata).
        dataset_name: Human-readable dataset name (stored as metadata).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target_fpr": target_fpr,
        "calibration_dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "detector_configs": detector_configs or {},
    }

    with path.open("w", encoding="utf-8") as fh:
        from tadbench.utils.io import json_safe
        json.dump(json_safe(payload), fh, indent=2, allow_nan=False)

    logger.info("save_thresholds: wrote %d thresholds to %s", len(thresholds), path)


def load_thresholds(path: Union[str, Path]) -> dict:
    """Load calibrated thresholds from a JSON file.

    Args:
        path: Path to a JSON file written by :func:`save_thresholds`.

    Returns:
        The full payload dict including metadata keys ``target_fpr``,
        ``calibration_dataset``, ``timestamp``, and ``thresholds``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"load_thresholds: file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"load_thresholds: invalid JSON in {path}: {exc}") from exc

    if "thresholds" not in payload:
        raise ValueError(
            f"load_thresholds: missing 'thresholds' key in {path}. "
            "File may not have been created by save_thresholds()."
        )

    payload["thresholds"] = {key: float(value) for key, value in payload["thresholds"].items()}
    for config in payload.get("detector_configs", {}).values():
        for key in list(config):
            if key.endswith("_threshold"):
                config[key] = float(config[key])
    logger.debug("load_thresholds: loaded %d thresholds from %s", len(payload["thresholds"]), path)
    return payload
