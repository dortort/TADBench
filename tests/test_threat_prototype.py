"""Tests for tadbench.detectors.threat_prototype.ThreatPrototypeDetector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tadbench.detectors.threat_prototype import ThreatPrototypeDetector


class TestThreatPrototypeFitScore:
    def test_fit_builds_index(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder)
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        assert det._index is not None
        assert det._index.ntotal == len(malicious)

    def test_fit_empty_raises(self, mock_embedder):
        det = ThreatPrototypeDetector(mock_embedder)
        with pytest.raises(ValueError, match="empty"):
            det.fit([])

    def test_score_returns_correct_shape(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder)
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        scores = det.score(sample_data)
        assert scores.shape == (len(sample_data),)

    def test_score_without_fit_raises(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.score(sample_data)

    def test_score_single_returns_float(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder)
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        score = det.score_single(sample_data[0])
        assert isinstance(score, float)

    def test_name_property(self, mock_embedder):
        det = ThreatPrototypeDetector(mock_embedder, index_type="flat", top_k=5, aggregation="mean")
        assert "threat_prototype" in det.name
        assert "flat" in det.name
        assert "5" in det.name
        assert "mean" in det.name


class TestThreatPrototypeAggregation:
    def _fit_and_score(self, mock_embedder, sample_data, aggregation, top_k=3):
        det = ThreatPrototypeDetector(mock_embedder, top_k=top_k, aggregation=aggregation)
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        return det.score(sample_data)

    def test_max_aggregation(self, mock_embedder, sample_data):
        scores = self._fit_and_score(mock_embedder, sample_data, "max", top_k=1)
        assert scores.shape == (len(sample_data),)

    def test_mean_aggregation(self, mock_embedder, sample_data):
        scores = self._fit_and_score(mock_embedder, sample_data, "mean", top_k=3)
        assert scores.shape == (len(sample_data),)

    def test_weighted_aggregation(self, mock_embedder, sample_data):
        scores = self._fit_and_score(mock_embedder, sample_data, "weighted", top_k=3)
        assert scores.shape == (len(sample_data),)

    def test_invalid_aggregation_raises(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder, aggregation="bad_agg")
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        with pytest.raises(ValueError, match="aggregation"):
            det.score(sample_data)


class TestThreatPrototypeIndexSaveLoad:
    def test_save_and_load_round_trip(self, mock_embedder, sample_data, tmp_path):
        det = ThreatPrototypeDetector(mock_embedder)
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        original_scores = det.score(sample_data)

        index_path = tmp_path / "test.index"
        det.save_index(index_path)

        assert index_path.exists()
        meta_path = tmp_path / "test.meta.json"
        assert meta_path.exists()

        det2 = ThreatPrototypeDetector(mock_embedder)
        det2.load_index(index_path)
        loaded_scores = det2.score(sample_data)

        np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=5)

    def test_metadata_json_contains_required_fields(self, mock_embedder, threat_bank_jsonl, tmp_path):
        # build_index() populates _metadata; fit() does not
        det = ThreatPrototypeDetector(mock_embedder)
        det.build_index(threat_bank_jsonl)
        index_path = tmp_path / "meta_test.index"
        det.save_index(index_path)

        meta_path = tmp_path / "meta_test.meta.json"
        meta = json.loads(meta_path.read_text())
        for key in ("embedder", "index_type", "sample_count", "dimension", "build_timestamp"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_build_index_from_jsonl(self, mock_embedder, threat_bank_jsonl, sample_data, tmp_path):
        det = ThreatPrototypeDetector(mock_embedder)
        det.build_index(threat_bank_jsonl)
        scores = det.score(sample_data)
        assert scores.shape == (len(sample_data),)

    def test_load_nonexistent_index_raises(self, mock_embedder, tmp_path):
        det = ThreatPrototypeDetector(mock_embedder)
        with pytest.raises(Exception):
            det.load_index(tmp_path / "nonexistent.index")


class TestThreatPrototypeIndexTypes:
    def test_flat_index(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder, index_type="flat")
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        scores = det.score(sample_data)
        assert len(scores) == len(sample_data)

    def test_hnsw_index(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder, index_type="hnsw")
        malicious = [s for s in sample_data if s.label == "malicious"]
        det.fit(malicious)
        scores = det.score(sample_data)
        assert len(scores) == len(sample_data)

    def test_hnsw_scores_match_flat_similarities(self, mock_embedder, sample_data):
        malicious = [s for s in sample_data if s.label == "malicious"]
        scores = {}
        for index_type in ("flat", "hnsw"):
            det = ThreatPrototypeDetector(
                mock_embedder, index_type=index_type, top_k=5, aggregation="mean"
            )
            det.fit(malicious)
            scores[index_type] = det.score(sample_data)

        np.testing.assert_allclose(scores["hnsw"], scores["flat"], atol=1e-5)

    def test_invalid_index_type_raises(self, mock_embedder, sample_data):
        det = ThreatPrototypeDetector(mock_embedder, index_type="bad_type")
        malicious = [s for s in sample_data if s.label == "malicious"]
        with pytest.raises(ValueError, match="index_type"):
            det.fit(malicious)
