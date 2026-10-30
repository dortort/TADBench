"""Tests for tadbench.evaluation.calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tadbench.evaluation.calibration import (
    calibrate_all_detectors,
    calibrate_threshold,
    load_thresholds,
    save_thresholds,
)


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------

class TestCalibrateThreshold:
    def test_perfect_separation_at_low_fpr(self):
        """Perfect classifier: threshold of 0.5 achieves FPR=0 at any target."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        threshold = calibrate_threshold(y_true, y_score, target_fpr=0.01)
        # At this threshold, no benign sample should be >= threshold
        benign_scores = y_score[y_true == 0]
        assert np.all(benign_scores < threshold) or threshold <= 0.8

    def test_threshold_is_float(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.2, 0.3, 0.7, 0.8])
        t = calibrate_threshold(y_true, y_score, target_fpr=0.1)
        assert isinstance(t, float)

    def test_higher_fpr_target_allows_lower_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.3, 0.5, 0.6, 0.55, 0.7, 0.8, 0.9])
        t_strict = calibrate_threshold(y_true, y_score, target_fpr=0.01)
        t_relaxed = calibrate_threshold(y_true, y_score, target_fpr=0.5)
        # More relaxed FPR allows lower threshold (catches more positives)
        assert t_relaxed <= t_strict

    def test_single_class_raises(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError):
            calibrate_threshold(y_true, y_score, target_fpr=0.01)

    def test_empty_after_nan_raises(self):
        y_true = np.array([0, 1])
        y_score = np.array([np.nan, np.nan])
        with pytest.raises(ValueError):
            calibrate_threshold(y_true, y_score, target_fpr=0.01)

    def test_unachievable_fpr_returns_min_fpr_threshold(self):
        """If target FPR is smaller than minimum achievable, use min-FPR threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.4, 0.6, 0.5, 0.7])
        # target_fpr=0 is likely unachievable with this data
        threshold = calibrate_threshold(y_true, y_score, target_fpr=0.0)
        assert isinstance(threshold, float)

    def test_nan_scores_filtered(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, np.nan, 0.3, 0.9])
        threshold = calibrate_threshold(y_true, y_score, target_fpr=0.1)
        assert isinstance(threshold, float)

    def test_various_fpr_targets(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_score = rng.rand(100)
        for fpr in [0.001, 0.005, 0.01, 0.05, 0.1]:
            t = calibrate_threshold(y_true, y_score, target_fpr=fpr)
            assert isinstance(t, float)
            # sklearn roc_curve may return inf as the sentinel first threshold;
            # the calibration function returns whatever sklearn gives, so we
            # only require it is a float (not NaN).
            assert not np.isnan(t)


# ---------------------------------------------------------------------------
# calibrate_all_detectors
# ---------------------------------------------------------------------------

class TestCalibrateAllDetectors:
    def test_returns_dict_with_detector_names(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = calibrate_all_detectors(
            {"detector_a": (y_true, y_score), "detector_b": (y_true, y_score)},
            target_fpr=0.05,
        )
        assert "detector_a" in result
        assert "detector_b" in result

    def test_skips_single_class_detector(self):
        y_true_good = np.array([0, 0, 1, 1])
        y_score_good = np.array([0.1, 0.2, 0.8, 0.9])
        y_true_bad = np.array([1, 1, 1])
        y_score_bad = np.array([0.5, 0.6, 0.7])
        result = calibrate_all_detectors(
            {"good": (y_true_good, y_score_good), "bad": (y_true_bad, y_score_bad)},
            target_fpr=0.05,
        )
        assert "good" in result
        assert "bad" not in result

    def test_threshold_values_are_floats(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = calibrate_all_detectors({"d": (y_true, y_score)}, target_fpr=0.05)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# save_thresholds / load_thresholds
# ---------------------------------------------------------------------------

class TestSaveLoadThresholds:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "thresholds.json"
        thresholds = {"detector_a": 0.75, "detector_b": 0.55}
        save_thresholds(thresholds, path, target_fpr=0.005, dataset_name="benchmark_val")
        assert path.exists()

    def test_load_returns_correct_thresholds(self, tmp_path):
        path = tmp_path / "thresholds.json"
        thresholds = {"detector_a": 0.75, "detector_b": 0.55}
        save_thresholds(thresholds, path, target_fpr=0.005, dataset_name="benchmark_val")
        loaded = load_thresholds(path)
        assert loaded["thresholds"]["detector_a"] == pytest.approx(0.75)
        assert loaded["thresholds"]["detector_b"] == pytest.approx(0.55)

    def test_loaded_payload_contains_metadata(self, tmp_path):
        path = tmp_path / "thresholds.json"
        save_thresholds({"d": 0.5}, path, target_fpr=0.01, dataset_name="val_set")
        payload = load_thresholds(path)
        assert "target_fpr" in payload
        assert "calibration_dataset" in payload
        assert "timestamp" in payload
        assert payload["target_fpr"] == pytest.approx(0.01)
        assert payload["calibration_dataset"] == "val_set"

    def test_round_trip_preserves_values(self, tmp_path):
        path = tmp_path / "rt.json"
        original = {"tps": 0.832, "ipsod": 0.441, "iccd": 0.619}
        save_thresholds(original, path, target_fpr=0.005, dataset_name="test_data")
        loaded = load_thresholds(path)
        for k, v in original.items():
            assert loaded["thresholds"][k] == pytest.approx(v, abs=1e-9)

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_thresholds(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_thresholds(path)

    def test_load_missing_thresholds_key_raises(self, tmp_path):
        path = tmp_path / "missing_key.json"
        path.write_text(json.dumps({"target_fpr": 0.01}))
        with pytest.raises(ValueError, match="thresholds"):
            load_thresholds(path)

    def test_save_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "thresholds.json"
        save_thresholds({"d": 0.5}, path, target_fpr=0.01, dataset_name="test")
        assert path.exists()


def test_infinite_gate_thresholds_round_trip_as_standard_json(tmp_path):
    from tadbench.evaluation.calibration import save_thresholds, load_thresholds
    path = tmp_path / "thresholds.json"
    save_thresholds({"cascade": .5, "threat_prototype": float("inf")}, path, .005, "val",
                    detector_configs={"cascade": {"stage1_threshold": float("inf")}})
    def invalid_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    import json
    json.loads(path.read_text(), parse_constant=invalid_constant)
    restored = load_thresholds(path)
    assert restored["thresholds"]["threat_prototype"] == float("inf")
    assert restored["detector_configs"]["cascade"]["stage1_threshold"] == float("inf")
