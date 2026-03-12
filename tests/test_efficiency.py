"""Tests for tadbench.evaluation.efficiency."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
from tadbench.evaluation.efficiency import (
    EfficiencyReport,
    LatencyProfiler,
    measure_index_build_time,
    measure_index_update_cost,
    measure_scaling_curves,
)


def _make_sample(
    body: str,
    tokens: int | None = None,
    label: str = "benign",
) -> BenchmarkSample:
    tok = tokens if tokens is not None else len(body.split())
    return BenchmarkSample(
        id=str(uuid.uuid4()),
        advertised_description="",
        prompt_body=body,
        label=label,
        attack_type="none" if label == "benign" else "direct_injection",
        source="deepset",
        metadata={"tokens": tok, "language": "en"},
    )


def _make_samples(n: int) -> list[BenchmarkSample]:
    bodies = [
        "This is a sample with some text content to process.",
        "Another text sample for testing purposes.",
    ]
    return [_make_sample(bodies[i % 2]) for i in range(n)]


def _make_fitted_detector(mock_embedder, n_threat: int = 5) -> ThreatPrototypeDetector:
    det = ThreatPrototypeDetector(mock_embedder)
    threats = [
        _make_sample(f"Malicious injection attack number {i}", label="malicious")
        for i in range(n_threat)
    ]
    det.fit(threats)
    return det


class TestLatencyProfiler:
    def test_returns_efficiency_report(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert isinstance(report, EfficiencyReport)

    def test_p50_nonzero(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(10)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert report.p50_ms >= 0.0

    def test_p95_geq_p50(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(20)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=2)
        assert report.p95_ms >= report.p50_ms

    def test_p99_geq_p95(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(20)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=2)
        assert report.p99_ms >= report.p95_ms

    def test_throughput_positive(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert report.throughput_rps > 0.0

    def test_peak_rss_nonnegative(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert report.peak_rss_mb >= 0.0

    def test_embedding_calls_per_sample_set(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert report.embedding_calls_per_sample >= 1.0

    def test_total_tokens_positive(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)
        assert report.total_tokens_embedded > 0

    def test_empty_samples_returns_zeros(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, [], n_runs=1)
        assert report.p50_ms == 0.0
        assert report.throughput_rps == 0.0

    def test_sampling_caps_latency_and_throughput_work(self):
        class CountingDetector:
            embedding_calls_per_sample = 1
            cross_encoder_calls_per_sample = 0

            def __init__(self):
                self.score_single_calls = 0
                self.batch_sizes: list[int] = []

            def score_single(self, sample):
                self.score_single_calls += 1
                return 0.0

            def score(self, samples):
                self.batch_sizes.append(len(samples))
                return np.zeros(len(samples))

        det = CountingDetector()
        samples = _make_samples(10)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(
            det,
            samples,
            n_runs=2,
            max_latency_samples=3,
            max_throughput_samples=4,
        )

        assert det.score_single_calls == 6
        assert det.batch_sizes == [4]
        assert report.latency_profile_samples == 3
        assert report.throughput_profile_samples == 4


class TestEfficiencyReportFields:
    def test_all_fields_present(self, mock_embedder):
        det = _make_fitted_detector(mock_embedder)
        samples = _make_samples(5)
        profiler = LatencyProfiler()
        report = profiler.profile_detector(det, samples, n_runs=1)

        assert hasattr(report, "p50_ms")
        assert hasattr(report, "p95_ms")
        assert hasattr(report, "p99_ms")
        assert hasattr(report, "throughput_rps")
        assert hasattr(report, "peak_rss_mb")
        assert hasattr(report, "cpu_utilization_pct")
        assert hasattr(report, "index_build_time_s")
        assert hasattr(report, "index_update_cost_s")
        assert hasattr(report, "embedding_calls_per_sample")
        assert hasattr(report, "total_tokens_embedded")
        assert hasattr(report, "cross_encoder_calls_per_sample")
        assert hasattr(report, "latency_profile_samples")
        assert hasattr(report, "throughput_profile_samples")


class TestMeasureIndexBuildTime:
    def test_returns_positive_float(self, mock_embedder):
        threats = [
            _make_sample(f"Threat sample {i}", label="malicious") for i in range(5)
        ]
        det = ThreatPrototypeDetector(mock_embedder)
        elapsed = measure_index_build_time(det, threats)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_detector_is_fitted_after_call(self, mock_embedder):
        threats = [_make_sample(f"Attack {i}", label="malicious") for i in range(5)]
        det = ThreatPrototypeDetector(mock_embedder)
        measure_index_build_time(det, threats)
        assert det._index is not None


class TestMeasureIndexUpdateCost:
    def test_returns_zero_without_add_samples(self, mock_embedder):
        threats = [_make_sample(f"Threat {i}", label="malicious") for i in range(5)]
        det = ThreatPrototypeDetector(mock_embedder)
        det.fit(threats)
        # ThreatPrototypeDetector has no add_samples() method
        cost = measure_index_update_cost(det, threats[:3], n_updates=3)
        assert cost == 0.0

    def test_empty_new_samples_returns_zero(self, mock_embedder):
        threats = [_make_sample(f"Threat {i}", label="malicious") for i in range(5)]
        det = ThreatPrototypeDetector(mock_embedder)
        det.fit(threats)
        cost = measure_index_update_cost(det, [], n_updates=5)
        assert cost == 0.0


class TestMeasureScalingCurves:
    def test_prompt_length_returns_list_of_dicts(self, mock_embedder):
        threats = [_make_sample(f"Threat {i}", label="malicious") for i in range(5)]
        det = ThreatPrototypeDetector(mock_embedder)
        det.fit(threats)

        # Build samples with controlled token counts
        short_samples = [_make_sample("hi", tokens=10) for _ in range(3)]
        long_samples = [_make_sample(" ".join(["word"] * 200), tokens=200) for _ in range(3)]
        all_samples = short_samples + long_samples

        results = measure_scaling_curves(det, all_samples, "prompt_length", [0, 128, 512])
        assert isinstance(results, list)
        for entry in results:
            assert "variable" in entry
            assert "variable_value" in entry
            assert "p50_ms" in entry
            assert "p95_ms" in entry

    def test_invalid_variable_raises(self, mock_embedder):
        threats = [_make_sample(f"Threat {i}", label="malicious") for i in range(5)]
        det = ThreatPrototypeDetector(mock_embedder)
        det.fit(threats)
        with pytest.raises(ValueError, match="Unknown scaling variable"):
            measure_scaling_curves(det, threats, "bad_variable", [0, 100])
