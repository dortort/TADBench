"""Efficiency profiling: latency, throughput, memory, CPU, scaling curves."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import psutil

if TYPE_CHECKING:
    from tadbench.data.schema import BenchmarkSample
    from tadbench.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

BenchmarkMode = Literal["full", "fast"]


@dataclass(frozen=True)
class ProfilingModeConfig:
    """Execution profile for efficiency measurements."""

    mode: BenchmarkMode
    latency_runs: int
    max_latency_samples: int | None
    max_throughput_samples: int | None


MATRIX_PROFILING_MODES: dict[BenchmarkMode, ProfilingModeConfig] = {
    "full": ProfilingModeConfig(
        mode="full",
        latency_runs=3,
        max_latency_samples=None,
        max_throughput_samples=None,
    ),
    "fast": ProfilingModeConfig(
        mode="fast",
        latency_runs=1,
        max_latency_samples=100,
        max_throughput_samples=250,
    ),
}


@dataclass
class EfficiencyReport:
    """Comprehensive efficiency measurements for a detector run."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    peak_rss_mb: float
    cpu_utilization_pct: float
    index_build_time_s: Optional[float] = None
    index_update_cost_s: Optional[float] = None
    embedding_calls_per_sample: float = 1.0
    total_tokens_embedded: int = 0
    cross_encoder_calls_per_sample: float = 0.0
    latency_profile_samples: int = 0
    throughput_profile_samples: int = 0


class LatencyProfiler:
    """Profile detector latency, throughput, and resource usage."""

    def profile_detector(
        self,
        detector: "BaseDetector",
        samples: list["BenchmarkSample"],
        n_runs: int = 3,
        max_latency_samples: int | None = None,
        max_throughput_samples: int | None = None,
    ) -> EfficiencyReport:
        """Profile a detector on the provided samples.

        Measures:
        - Per-sample latency via ``score_single()`` (p50/p95/p99), repeated n_runs times.
        - Throughput via ``score()`` on the full batch.
        - Peak RSS delta via psutil before/after the batch score call.
        - CPU utilization via psutil during the batch score call.

        Args:
            detector: Fitted detector implementing BaseDetector.
            samples: Samples to score (should already have been seen in fit if required).
            n_runs: Number of repetition runs for latency measurement.
            max_latency_samples: Optional cap on samples used for per-sample
                latency profiling.
            max_throughput_samples: Optional cap on samples used for batch
                throughput/RSS/CPU measurement.

        Returns:
            EfficiencyReport with all measured fields populated.
        """
        if not samples:
            logger.warning("LatencyProfiler.profile_detector: empty samples list")
            return EfficiencyReport(
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                throughput_rps=0.0,
                peak_rss_mb=0.0,
                cpu_utilization_pct=0.0,
                latency_profile_samples=0,
                throughput_profile_samples=0,
            )

        process = psutil.Process()
        latency_samples = samples[:max_latency_samples] if max_latency_samples is not None else samples
        throughput_samples = (
            samples[:max_throughput_samples]
            if max_throughput_samples is not None
            else samples
        )

        # --- Per-sample latency via score_single() ---
        all_latencies_ms: list[float] = []
        for _ in range(n_runs):
            for sample in latency_samples:
                t0 = time.perf_counter()
                detector.score_single(sample)
                t1 = time.perf_counter()
                all_latencies_ms.append((t1 - t0) * 1000.0)

        latencies = np.array(all_latencies_ms, dtype=float)
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        # --- Throughput and RSS delta via batch score() ---
        rss_before = process.memory_info().rss

        # Prime CPU percent measurement (first call returns 0.0 for the interval).
        process.cpu_percent(interval=None)

        batch_t0 = time.perf_counter()
        detector.score(throughput_samples)
        batch_elapsed = time.perf_counter() - batch_t0

        cpu_pct = process.cpu_percent(interval=None)
        rss_after = process.memory_info().rss

        throughput = len(throughput_samples) / batch_elapsed if batch_elapsed > 0 else 0.0
        peak_rss_mb = max(0.0, (rss_after - rss_before) / (1024 * 1024))

        # --- Cost proxies ---
        embedding_calls = float(getattr(detector, "embedding_calls_per_sample", 1))
        cross_encoder_calls = float(
            getattr(detector, "cross_encoder_calls_per_sample", 0)
        )

        # Approximate total tokens: sum of whitespace-token counts in prompt bodies.
        total_tokens = sum(
            len(s.prompt_body.split()) for s in samples
        ) * int(embedding_calls)

        return EfficiencyReport(
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            throughput_rps=throughput,
            peak_rss_mb=peak_rss_mb,
            cpu_utilization_pct=float(cpu_pct),
            embedding_calls_per_sample=embedding_calls,
            total_tokens_embedded=total_tokens,
            cross_encoder_calls_per_sample=cross_encoder_calls,
            latency_profile_samples=len(latency_samples),
            throughput_profile_samples=len(throughput_samples),
        )


def resolve_matrix_profiling_mode(mode: BenchmarkMode) -> ProfilingModeConfig:
    """Return the profiling configuration used by runner-backed benchmarks."""
    return MATRIX_PROFILING_MODES[mode]


def measure_index_build_time(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
) -> float:
    """Time how long detector.fit() takes.

    Args:
        detector: An unfitted detector.
        samples: Training samples (e.g. threat bank entries).

    Returns:
        Elapsed time in seconds.
    """
    t0 = time.perf_counter()
    detector.fit(samples)
    elapsed = time.perf_counter() - t0
    logger.debug("measure_index_build_time: %.3fs for %d samples", elapsed, len(samples))
    return float(elapsed)


def measure_index_update_cost(
    detector: "BaseDetector",
    new_samples: list["BenchmarkSample"],
    n_updates: int = 10,
) -> float:
    """Measure average cost to incrementally add samples to an existing index.

    Attempts to call ``detector.add_samples()`` if available; falls back to
    re-fitting on the combined sample set. Only meaningful for TPS-style detectors
    that support incremental FAISS updates.

    Args:
        detector: A previously fitted detector.
        new_samples: Pool of samples to add incrementally. At most n_updates are used.
        n_updates: Number of individual add operations to average over.

    Returns:
        Average time per update in seconds. Returns 0.0 if not supported.
    """
    samples_to_add = new_samples[:n_updates]
    if not samples_to_add:
        return 0.0

    add_fn = getattr(detector, "add_samples", None)
    if add_fn is not None:
        times: list[float] = []
        for sample in samples_to_add:
            t0 = time.perf_counter()
            add_fn([sample])
            times.append(time.perf_counter() - t0)
        avg = float(np.mean(times)) if times else 0.0
        logger.debug(
            "measure_index_update_cost: avg %.4fs over %d updates", avg, len(times)
        )
        return avg

    # Fallback: re-fit is the only way to "update" — not truly incremental.
    logger.warning(
        "measure_index_update_cost: detector %s has no add_samples(); "
        "returning 0.0 (update cost undefined)",
        getattr(detector, "name", type(detector).__name__),
    )
    return 0.0


def measure_scaling_curves(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
    variable: str,
    values: list,
) -> list[dict]:
    """Measure how latency scales with a chosen variable.

    Supported variables:
    - ``"prompt_length"``: Filter samples into token-count bins defined by ``values``
      (list of (low, high) tuples or bin edge integers); measure p50/p95 per bin.
    - ``"threat_bank_size"``: Sub-sample ``samples`` to each size in ``values``
      (e.g. [100, 500, 1000, 5000, 10000]), re-fit detector, measure query latency.
    - ``"chunks_per_prompt"``: Group samples by approximate chunk count (word-count
      proxy) matching ``values`` bin edges; measure latency per group.
    - ``"pair_length"``: Group by combined description+body token count matching
      ``values`` bin edges; measure latency per bin.

    Args:
        detector: A fitted (or fittable) detector.
        samples: Full sample pool.
        variable: One of the supported variable names.
        values: Variable-specific configuration (bin edges or size targets).

    Returns:
        List of dicts, each with keys: ``variable``, ``variable_value``,
        ``n_samples``, ``p50_ms``, ``p95_ms``.
    """
    results: list[dict] = []

    if variable == "prompt_length":
        results = _scaling_prompt_length(detector, samples, values)
    elif variable == "threat_bank_size":
        results = _scaling_threat_bank_size(detector, samples, values)
    elif variable == "chunks_per_prompt":
        results = _scaling_chunks_per_prompt(detector, samples, values)
    elif variable == "pair_length":
        results = _scaling_pair_length(detector, samples, values)
    else:
        raise ValueError(
            f"Unknown scaling variable '{variable}'. "
            "Supported: prompt_length, threat_bank_size, chunks_per_prompt, pair_length"
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _measure_latencies(
    detector: "BaseDetector",
    subset: list["BenchmarkSample"],
) -> tuple[float, float]:
    """Return (p50_ms, p95_ms) for score_single() on subset."""
    if not subset:
        return 0.0, 0.0
    latencies_ms: list[float] = []
    for s in subset:
        t0 = time.perf_counter()
        detector.score_single(s)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies_ms, dtype=float)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _scaling_prompt_length(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
    values: list,
) -> list[dict]:
    """Latency vs prompt token-count bins.

    ``values`` should be a list of bin-edge integers, e.g. [0, 128, 512, 2048, 99999].
    Adjacent pairs define bins: [v[i], v[i+1]).
    """
    if len(values) < 2:
        logger.warning("_scaling_prompt_length: need at least 2 bin edges in values")
        return []

    results: list[dict] = []
    edges = sorted(values)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        subset = [
            s for s in samples
            if lo <= s.metadata.get("tokens", len(s.prompt_body.split())) < hi
        ]
        if not subset:
            continue
        p50, p95 = _measure_latencies(detector, subset)
        results.append({
            "variable": "prompt_length",
            "variable_value": f"{lo}-{hi}",
            "n_samples": len(subset),
            "p50_ms": p50,
            "p95_ms": p95,
        })
    return results


def _scaling_threat_bank_size(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
    values: list,
) -> list[dict]:
    """Latency vs threat bank size.

    ``values`` should be a list of integers, e.g. [100, 500, 1000, 5000, 10000].
    For each size, sub-samples ``samples`` to that count, re-fits the detector,
    then measures query latency on the *full* sample set (to keep query set stable).
    """
    results: list[dict] = []
    rng = np.random.default_rng(42)

    # Keep the query set separate from the bank — use the full sample set as queries.
    query_samples = samples

    for size in sorted(values):
        if size > len(samples):
            logger.debug(
                "_scaling_threat_bank_size: requested size %d > available %d; skipping",
                size,
                len(samples),
            )
            continue
        indices = rng.choice(len(samples), size=size, replace=False)
        bank_subset = [samples[i] for i in indices]

        t_fit0 = time.perf_counter()
        detector.fit(bank_subset)
        fit_time = time.perf_counter() - t_fit0

        p50, p95 = _measure_latencies(detector, query_samples)
        results.append({
            "variable": "threat_bank_size",
            "variable_value": size,
            "n_samples": len(query_samples),
            "fit_time_s": float(fit_time),
            "p50_ms": p50,
            "p95_ms": p95,
        })
    return results


def _scaling_chunks_per_prompt(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
    values: list,
) -> list[dict]:
    """Latency vs approximate chunk count per prompt.

    ``values`` should be bin-edge integers based on approximate chunk count
    (estimated as word-count // 128, clamped to at least 1).
    """
    if len(values) < 2:
        logger.warning("_scaling_chunks_per_prompt: need at least 2 bin edges in values")
        return []

    def _approx_chunks(s: "BenchmarkSample") -> int:
        words = len(s.prompt_body.split())
        return max(1, words // 128)

    results: list[dict] = []
    edges = sorted(values)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        subset = [s for s in samples if lo <= _approx_chunks(s) < hi]
        if not subset:
            continue
        p50, p95 = _measure_latencies(detector, subset)
        results.append({
            "variable": "chunks_per_prompt",
            "variable_value": f"{lo}-{hi}",
            "n_samples": len(subset),
            "p50_ms": p50,
            "p95_ms": p95,
        })
    return results


def _scaling_pair_length(
    detector: "BaseDetector",
    samples: list["BenchmarkSample"],
    values: list,
) -> list[dict]:
    """Latency vs combined description+body token count.

    ``values`` should be bin-edge integers, e.g. [0, 256, 512, 1024, 4096].
    """
    if len(values) < 2:
        logger.warning("_scaling_pair_length: need at least 2 bin edges in values")
        return []

    def _pair_tokens(s: "BenchmarkSample") -> int:
        desc_tokens = len(s.advertised_description.split())
        body_tokens = s.metadata.get("tokens", len(s.prompt_body.split()))
        return desc_tokens + body_tokens

    results: list[dict] = []
    edges = sorted(values)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        subset = [s for s in samples if lo <= _pair_tokens(s) < hi]
        if not subset:
            continue
        p50, p95 = _measure_latencies(detector, subset)
        results.append({
            "variable": "pair_length",
            "variable_value": f"{lo}-{hi}",
            "n_samples": len(subset),
            "p50_ms": p50,
            "p95_ms": p95,
        })
    return results
