"""Drift evaluation: measure recall degradation when a new attack family appears."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tadbench.evaluation.calibration import calibrate_threshold

if TYPE_CHECKING:
    from tadbench.data.schema import BenchmarkSample
    from tadbench.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

# Minimum number of holdout-class samples required to produce a meaningful recall.
_MIN_HOLDOUT_SAMPLES = 10


@dataclass
class DriftReport:
    """Results of a drift evaluation for a single held-out attack type."""

    holdout_attack_type: str
    recall_on_known_attacks: float
    """Recall computed only on attack types seen during training."""
    recall_on_holdout_attack: float
    """Recall computed only on the held-out (unseen) attack type."""
    recall_degradation: float
    """recall_on_known_attacks - recall_on_holdout_attack (positive = degradation)."""
    threshold_before: float
    """Threshold calibrated on val set without the holdout attack type."""
    threshold_after_retrain: float
    """Threshold re-calibrated after adding holdout examples to training data."""
    threshold_drift: float
    """Absolute difference |threshold_after_retrain - threshold_before|."""
    retraining_cost_seconds: float
    """Time in seconds to re-fit + re-calibrate using the extended training set."""


class DriftEvaluator:
    """Evaluate how well a detector generalises to an unseen attack family.

    Usage::

        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        train, val, test = evaluator.create_drift_splits(all_samples)
        report = evaluator.evaluate_drift(detector, train, val, test)
    """

    def __init__(self, holdout_attack_type: str) -> None:
        """Initialise the evaluator.

        Args:
            holdout_attack_type: The ``attack_type`` value to hold out from
                training and validation splits.  Examples appearing in the test
                set with this type serve as the unseen-attack probe.
        """
        self.holdout_attack_type = holdout_attack_type

    def create_drift_splits(
        self,
        all_samples: list["BenchmarkSample"],
    ) -> tuple[list["BenchmarkSample"], list["BenchmarkSample"], list["BenchmarkSample"]]:
        """Partition samples into train/val/test splits for drift evaluation.

        - **train_without_holdout**: All samples whose ``attack_type`` is NOT
          the holdout type (used to fit the detector).
        - **val_without_holdout**: Validation subset (15 % of non-holdout,
          stratified by label) for threshold calibration.
        - **test_with_holdout**: The remaining 15 % test split from
          non-holdout samples, *plus all* samples whose ``attack_type`` equals
          the holdout type.

        The 70/15/15 train/val/test split is stratified by label to maintain
        class balance, using ``seed=42`` for reproducibility.

        Args:
            all_samples: Complete sample pool.

        Returns:
            (train_without_holdout, val_without_holdout, test_with_holdout)
        """
        # Separate holdout samples from the rest.
        holdout_samples = [
            s for s in all_samples if s.attack_type == self.holdout_attack_type
        ]
        non_holdout = [
            s for s in all_samples if s.attack_type != self.holdout_attack_type
        ]

        if len(holdout_samples) < _MIN_HOLDOUT_SAMPLES:
            logger.warning(
                "DriftEvaluator: only %d samples of attack_type='%s' found (minimum: %d). "
                "Drift recall estimate may be unreliable.",
                len(holdout_samples),
                self.holdout_attack_type,
                _MIN_HOLDOUT_SAMPLES,
            )

        # Stratified 70/15/15 split of non-holdout samples.
        rng = np.random.default_rng(42)
        benign = [s for s in non_holdout if s.label == "benign"]
        malicious = [s for s in non_holdout if s.label == "malicious"]

        def _split_stratum(
            stratum: list["BenchmarkSample"],
        ) -> tuple[list, list, list]:
            """Split a single class into 70/15/15."""
            indices = rng.permutation(len(stratum)).tolist()
            n = len(stratum)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            train_idx = indices[:n_train]
            val_idx = indices[n_train: n_train + n_val]
            test_idx = indices[n_train + n_val:]
            return (
                [stratum[i] for i in train_idx],
                [stratum[i] for i in val_idx],
                [stratum[i] for i in test_idx],
            )

        b_train, b_val, b_test = _split_stratum(benign)
        m_train, m_val, m_test = _split_stratum(malicious)

        train_without_holdout = b_train + m_train
        val_without_holdout = b_val + m_val
        test_base = b_test + m_test

        # Test set includes both the base test portion AND the holdout attack samples.
        test_with_holdout = test_base + holdout_samples

        logger.debug(
            "DriftEvaluator.create_drift_splits: train=%d, val=%d, test=%d "
            "(base=%d, holdout=%d)",
            len(train_without_holdout),
            len(val_without_holdout),
            len(test_with_holdout),
            len(test_base),
            len(holdout_samples),
        )
        return train_without_holdout, val_without_holdout, test_with_holdout

    def evaluate_drift(
        self,
        detector: "BaseDetector",
        train: list["BenchmarkSample"],
        val: list["BenchmarkSample"],
        test: list["BenchmarkSample"],
        target_fpr: float = 0.005,
    ) -> DriftReport:
        """Run the full drift evaluation pipeline.

        Steps:
        1. Fit detector on ``train`` (no holdout attack type).
        2. Score ``val`` and calibrate threshold at ``target_fpr``.
        3. Score ``test`` (includes holdout attack samples).
        4. Compute recall on known attacks and on the holdout attack.
        5. Re-fit on ``train + holdout_train`` and re-calibrate to measure
           threshold drift and retraining cost.

        Args:
            detector: Unfitted detector instance.
            train: Training samples without the holdout attack type.
            val: Validation samples without the holdout attack type.
            test: Test samples including holdout attack samples.
            target_fpr: FPR used for threshold calibration.

        Returns:
            DriftReport with all metrics populated.
        """
        # --- Step 1: Fit on training data (no holdout). ---
        detector.fit(train)

        # --- Step 2: Calibrate threshold on validation set. ---
        val_scores = detector.score(val)
        val_labels = np.array(
            [1 if s.label == "malicious" else 0 for s in val], dtype=int
        )
        threshold_before = _safe_calibrate(val_scores, val_labels, target_fpr)

        # --- Step 3: Score test set. ---
        test_scores = detector.score(test)
        test_labels = np.array(
            [1 if s.label == "malicious" else 0 for s in test], dtype=int
        )

        # --- Step 4: Recall on known vs holdout attacks. ---
        # "Known attacks" = malicious samples NOT of the holdout type.
        known_mask = np.array(
            [
                s.label == "malicious" and s.attack_type != self.holdout_attack_type
                for s in test
            ],
            dtype=bool,
        )
        holdout_mask = np.array(
            [s.attack_type == self.holdout_attack_type for s in test],
            dtype=bool,
        )

        recall_known = _recall_at_threshold(
            test_labels[known_mask], test_scores[known_mask], threshold_before
        )
        recall_holdout = _recall_at_threshold(
            test_labels[holdout_mask], test_scores[holdout_mask], threshold_before
        )
        recall_degradation = recall_known - recall_holdout

        # --- Step 5: Retrain with holdout data and measure threshold drift. ---
        # Use the holdout samples from the test set as the "newly arrived" data.
        holdout_samples = [s for s in test if s.attack_type == self.holdout_attack_type]
        extended_train = train + holdout_samples

        retrain_t0 = time.perf_counter()
        detector.fit(extended_train)
        retrain_val_scores = detector.score(val)
        threshold_after_retrain = _safe_calibrate(
            retrain_val_scores, val_labels, target_fpr
        )
        retraining_cost_seconds = float(time.perf_counter() - retrain_t0)

        threshold_drift = abs(threshold_after_retrain - threshold_before)

        logger.info(
            "DriftEvaluator.evaluate_drift: holdout='%s' "
            "recall_known=%.3f recall_holdout=%.3f degradation=%.3f "
            "threshold_drift=%.4f retrain_cost=%.2fs",
            self.holdout_attack_type,
            recall_known,
            recall_holdout,
            recall_degradation,
            threshold_drift,
            retraining_cost_seconds,
        )

        return DriftReport(
            holdout_attack_type=self.holdout_attack_type,
            recall_on_known_attacks=recall_known,
            recall_on_holdout_attack=recall_holdout,
            recall_degradation=recall_degradation,
            threshold_before=threshold_before,
            threshold_after_retrain=threshold_after_retrain,
            threshold_drift=threshold_drift,
            retraining_cost_seconds=retraining_cost_seconds,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _recall_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> float:
    """Compute recall (TPR) of the given threshold on the provided subset.

    Returns 0.0 if there are no positive samples or no valid scores.
    """
    valid = ~np.isnan(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0

    predicted_positive = y_score >= threshold
    tp = int((predicted_positive & (y_true == 1)).sum())
    fn = int(((~predicted_positive) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _safe_calibrate(
    y_score: np.ndarray,
    y_true: np.ndarray,
    target_fpr: float,
) -> float:
    """Calibrate threshold with graceful fallback on single-class data."""
    try:
        return calibrate_threshold(y_true, y_score, target_fpr)
    except ValueError as exc:
        logger.warning("_safe_calibrate: calibration failed (%s); using score median", exc)
        valid = y_score[~np.isnan(y_score)]
        return float(np.median(valid)) if len(valid) > 0 else 0.5
