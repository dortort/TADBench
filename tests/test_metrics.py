"""Tests for tadbench.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tadbench.evaluation.metrics import (
    compute_all_metrics,
    compute_auprc,
    compute_auroc,
    compute_calibration_error,
    compute_per_group_metrics,
    compute_tpr_at_fpr,
)


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------

class TestComputeAuroc:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        assert compute_auroc(y_true, y_score) == pytest.approx(1.0, abs=1e-6)

    def test_inverted_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.8, 0.9, 1.0, 0.1, 0.2, 0.3])
        assert compute_auroc(y_true, y_score) == pytest.approx(0.0, abs=1e-6)

    def test_random_classifier_near_half(self):
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.randint(0, 2, n)
        y_score = rng.rand(n)
        auroc = compute_auroc(y_true, y_score)
        assert 0.4 < auroc < 0.6

    def test_single_class_returns_zero(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        assert compute_auroc(y_true, y_score) == 0.0

    def test_nan_scores_filtered(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, np.nan, 0.2, 0.9])
        # After filtering NaN: y_true=[0,0,1], y_score=[0.1,0.2,0.9]
        result = compute_auroc(y_true, y_score)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_all_nan_returns_zero(self):
        y_true = np.array([0, 1])
        y_score = np.array([np.nan, np.nan])
        assert compute_auroc(y_true, y_score) == 0.0


# ---------------------------------------------------------------------------
# compute_auprc
# ---------------------------------------------------------------------------

class TestComputeAuprc:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auprc(y_true, y_score) == pytest.approx(1.0, abs=1e-6)

    def test_single_class_returns_zero(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.5, 0.6, 0.7])
        assert compute_auprc(y_true, y_score) == 0.0

    def test_reasonable_range(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        result = compute_auprc(y_true, y_score)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_tpr_at_fpr
# ---------------------------------------------------------------------------

class TestComputeTprAtFpr:
    def test_perfect_classifier_tpr_is_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_tpr_at_fpr(y_true, y_score, target_fprs=[0.01, 0.05, 0.1])
        for fpr, tpr in result.items():
            assert tpr == pytest.approx(1.0, abs=1e-6)

    def test_default_fprs_present(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_tpr_at_fpr(y_true, y_score)
        assert 0.001 in result
        assert 0.005 in result
        assert 0.01 in result

    def test_single_class_returns_zeros(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        result = compute_tpr_at_fpr(y_true, y_score, [0.01])
        assert result[0.01] == 0.0

    def test_tpr_values_in_range(self):
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, 100)
        y_score = rng.rand(100)
        result = compute_tpr_at_fpr(y_true, y_score, [0.01, 0.05, 0.1])
        for tpr in result.values():
            assert 0.0 <= tpr <= 1.0

    def test_tpr_increases_with_fpr_target(self):
        rng = np.random.RandomState(2)
        y_true = (rng.rand(200) > 0.5).astype(int)
        y_score = y_true * 0.6 + rng.rand(200) * 0.4
        result = compute_tpr_at_fpr(y_true, y_score, [0.01, 0.05, 0.1])
        # TPR at higher FPR target should be >= TPR at lower FPR target
        assert result[0.05] >= result[0.01] - 0.05  # allow small tolerance
        assert result[0.1] >= result[0.05] - 0.05


# ---------------------------------------------------------------------------
# compute_calibration_error
# ---------------------------------------------------------------------------

class TestComputeCalibrationError:
    def test_perfect_calibration_returns_zero(self):
        # If scores exactly match the positive rates, ECE = 0
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        # All same score — after normalization, ECE = 0 (all scores identical path)
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ece = compute_calibration_error(y_true, y_score)
        assert ece == 0.0

    def test_result_in_range(self):
        rng = np.random.RandomState(3)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        ece = compute_calibration_error(y_true, y_score)
        assert 0.0 <= ece <= 1.0

    def test_nan_scores_handled(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, np.nan, 0.3, 0.9])
        ece = compute_calibration_error(y_true, y_score)
        assert np.isfinite(ece)

    def test_empty_after_nan_filtering(self):
        y_true = np.array([0, 1])
        y_score = np.array([np.nan, np.nan])
        ece = compute_calibration_error(y_true, y_score)
        assert ece == 0.0


# ---------------------------------------------------------------------------
# compute_per_group_metrics
# ---------------------------------------------------------------------------

class TestComputePerGroupMetrics:
    def test_returns_dict_per_group(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        groups = ["A", "A", "A", "B", "B", "B"]
        result = compute_per_group_metrics(y_true, y_score, groups)
        assert "A" in result
        assert "B" in result

    def test_group_contains_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        groups = ["attack", "attack", "benign", "benign"]
        result = compute_per_group_metrics(y_true, y_score, groups)
        for group_name, metrics in result.items():
            assert "auroc" in metrics
            assert "auprc" in metrics
            assert "tpr_at_fpr" in metrics
            assert "n_samples" in metrics
            assert "n_positive" in metrics

    def test_all_positive_group_is_compared_against_all_negatives(self):
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.05])
        groups = ["none", "none", "A", "A", "B", "B"]
        result = compute_per_group_metrics(y_true, y_score, groups)
        assert result["A"]["auroc"] == pytest.approx(1.0)
        assert result["B"]["auroc"] == pytest.approx(0.25)
        assert result["A"]["n_samples"] == 2
        assert result["A"]["n_positive"] == 2
        assert result["A"]["tpr_at_fpr"][0.01] == pytest.approx(1.0)

    def test_group_without_positives_reports_nan(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        groups = ["none", "none", "A", "A"]
        result = compute_per_group_metrics(y_true, y_score, groups)
        assert np.isnan(result["none"]["auroc"])
        assert np.isnan(result["none"]["auprc"])
        assert all(np.isnan(v) for v in result["none"]["tpr_at_fpr"].values())

    def test_mixed_label_group_uses_only_its_own_samples(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.9, 0.1, 0.1, 0.9])
        groups = ["short", "short", "long", "long"]
        result = compute_per_group_metrics(y_true, y_score, groups)
        assert result["short"]["auroc"] == pytest.approx(0.0)
        assert result["long"]["auroc"] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            compute_per_group_metrics(
                np.array([0, 1]),
                np.array([0.1, 0.9, 0.5]),
                ["A", "B"],
            )


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_required_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_all_metrics(y_true, y_score)
        for key in ("auroc", "auprc", "tpr_at_fpr", "calibration_error", "n_samples", "n_positive"):
            assert key in result

    def test_with_groups_includes_per_group(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        groups = ["none", "direct_injection", "none", "jailbreak"]
        result = compute_all_metrics(y_true, y_score, groups=groups)
        assert "per_group" in result
        assert "none" in result["per_group"]

    def test_without_groups_no_per_group_key(self):
        y_true = np.array([0, 1])
        y_score = np.array([0.1, 0.9])
        result = compute_all_metrics(y_true, y_score)
        assert "per_group" not in result

    def test_perfect_classifier_auroc_1(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.array([0.1] * 50 + [0.9] * 50)
        result = compute_all_metrics(y_true, y_score)
        assert result["auroc"] == pytest.approx(1.0, abs=1e-6)
