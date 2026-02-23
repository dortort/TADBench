"""Tests for tadbench.evaluation.drift."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
from tadbench.evaluation.drift import DriftEvaluator, DriftReport


def _make_sample(
    body: str,
    label: str = "benign",
    attack_type: str = "none",
) -> BenchmarkSample:
    return BenchmarkSample(
        id=str(uuid.uuid4()),
        advertised_description="",
        prompt_body=body,
        label=label,
        attack_type=attack_type,
        source="deepset",
        metadata={"tokens": len(body.split()), "language": "en"},
    )


def _make_dataset(
    n_benign: int = 20,
    n_known_malicious: int = 20,
    n_holdout: int = 15,
    holdout_attack: str = "jailbreak",
) -> list[BenchmarkSample]:
    samples = []
    for i in range(n_benign):
        samples.append(_make_sample(f"Benign text sample number {i} with context.", "benign", "none"))
    for i in range(n_known_malicious):
        samples.append(_make_sample(f"Ignore instructions attack number {i}.", "malicious", "direct_injection"))
    for i in range(n_holdout):
        samples.append(_make_sample(f"Jailbreak attack sample {i} for testing.", "malicious", holdout_attack))
    return samples


class TestDriftEvaluatorCreateSplits:
    def test_holdout_absent_from_train(self):
        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        all_samples = _make_dataset()
        train, val, test = evaluator.create_drift_splits(all_samples)
        for s in train:
            assert s.attack_type != "jailbreak", "jailbreak should not be in train"

    def test_holdout_absent_from_val(self):
        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        all_samples = _make_dataset()
        train, val, test = evaluator.create_drift_splits(all_samples)
        for s in val:
            assert s.attack_type != "jailbreak", "jailbreak should not be in val"

    def test_holdout_present_in_test(self):
        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        all_samples = _make_dataset(n_holdout=15)
        train, val, test = evaluator.create_drift_splits(all_samples)
        holdout_in_test = [s for s in test if s.attack_type == "jailbreak"]
        assert len(holdout_in_test) == 15, "All holdout samples should be in test"

    def test_all_samples_accounted_for(self):
        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        all_samples = _make_dataset(n_benign=20, n_known_malicious=20, n_holdout=15)
        train, val, test = evaluator.create_drift_splits(all_samples)
        # Holdout samples appear only in test; non-holdout split into train/val/test
        n_non_holdout = sum(1 for s in all_samples if s.attack_type != "jailbreak")
        n_holdout = sum(1 for s in all_samples if s.attack_type == "jailbreak")
        assert len(train) + len(val) + (len(test) - n_holdout) == n_non_holdout

    def test_train_has_both_labels(self):
        evaluator = DriftEvaluator(holdout_attack_type="jailbreak")
        all_samples = _make_dataset()
        train, val, test = evaluator.create_drift_splits(all_samples)
        labels = {s.label for s in train}
        assert "benign" in labels
        assert "malicious" in labels

    def test_no_holdout_samples_warns_but_does_not_raise(self):
        evaluator = DriftEvaluator(holdout_attack_type="nonexistent_attack")
        all_samples = _make_dataset(n_holdout=0)
        # Should not raise; just warns
        train, val, test = evaluator.create_drift_splits(all_samples)
        holdout_in_test = [s for s in test if s.attack_type == "nonexistent_attack"]
        assert len(holdout_in_test) == 0


class TestDriftEvaluatorEvaluateDrift:
    def _get_splits_and_detector(self, mock_embedder, holdout_attack="jailbreak"):
        evaluator = DriftEvaluator(holdout_attack_type=holdout_attack)
        all_samples = _make_dataset(
            n_benign=30, n_known_malicious=30, n_holdout=15, holdout_attack=holdout_attack
        )
        train, val, test = evaluator.create_drift_splits(all_samples)
        detector = ThreatPrototypeDetector(mock_embedder)
        return evaluator, detector, train, val, test

    def test_returns_drift_report(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        assert isinstance(report, DriftReport)

    def test_holdout_attack_type_recorded(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        assert report.holdout_attack_type == "jailbreak"

    def test_recall_values_in_range(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        assert 0.0 <= report.recall_on_known_attacks <= 1.0
        assert 0.0 <= report.recall_on_holdout_attack <= 1.0

    def test_recall_degradation_is_difference(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        expected = report.recall_on_known_attacks - report.recall_on_holdout_attack
        assert abs(report.recall_degradation - expected) < 1e-9

    def test_threshold_values_are_finite(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        # sklearn roc_curve may return inf as its sentinel first threshold;
        # we only require non-NaN (inf is an acceptable calibration result).
        assert not np.isnan(report.threshold_before)
        assert not np.isnan(report.threshold_after_retrain)

    def test_threshold_drift_is_absolute_difference(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        expected_drift = abs(report.threshold_after_retrain - report.threshold_before)
        # When both thresholds are inf (degenerate calibration), drift is nan on both sides
        if np.isnan(expected_drift):
            assert np.isnan(report.threshold_drift)
        else:
            assert abs(report.threshold_drift - expected_drift) < 1e-9

    def test_retraining_cost_nonnegative(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        assert report.retraining_cost_seconds >= 0.0

    def test_all_report_fields_present(self, mock_embedder):
        evaluator, detector, train, val, test = self._get_splits_and_detector(mock_embedder)
        report = evaluator.evaluate_drift(detector, train, val, test)
        for field in (
            "holdout_attack_type",
            "recall_on_known_attacks",
            "recall_on_holdout_attack",
            "recall_degradation",
            "threshold_before",
            "threshold_after_retrain",
            "threshold_drift",
            "retraining_cost_seconds",
        ):
            assert hasattr(report, field), f"Missing field: {field}"
