"""Technique 2: Intra-Prompt Semantic Outlier Detection (IPSOD)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.base import BaseDetector
from tadbench.embedders.protocol import EmbedderProtocol
from tadbench.utils.chunker import get_chunker

logger = logging.getLogger(__name__)

OutlierDetectorName = Literal["lof", "iforest", "ecod"]
AggregationType = Literal["max", "top2_mean", "pct_above_threshold", "zscore_max"]
ChunkerName = Literal["sentence", "fixed_128"]

_BATCH_SIZE = 100  # Process this many samples at once to limit memory pressure


class IntraPromptOutlierDetector(BaseDetector):
    """Detect threats by finding semantically outlying chunks within a single prompt.

    Each prompt is split into chunks; an outlier detector is fitted on that
    prompt's own chunk embeddings.  The per-chunk anomaly scores are then
    aggregated into a single prompt-level score (higher = more malicious).

    fit() is a no-op because detection is performed per-prompt at score time.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        outlier_detector: OutlierDetectorName = "lof",
        chunker: ChunkerName = "sentence",
        aggregation: AggregationType = "max",
        min_chunks: int = 5,
        **kwargs: object,
    ) -> None:
        super().__init__(embedder, **kwargs)
        self.outlier_detector_name = outlier_detector
        self.chunker_name = chunker
        self.aggregation = aggregation
        self.min_chunks = min_chunks

        self._chunker = get_chunker(chunker)
        self._fallback_chunker = get_chunker("fixed_128")

        # Running average for embedding_calls_per_sample property
        self._avg_chunks: float = 1.0
        self._scored_count: int = 0

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            f"intra_prompt_outlier_{self.outlier_detector_name}"
            f"_{self.chunker_name}_{self.aggregation}"
        )

    @property
    def embedding_calls_per_sample(self) -> int:
        """Average number of embedding calls per sample (= average chunk count)."""
        return max(1, round(self._avg_chunks))

    def fit(self, samples: list[BenchmarkSample]) -> None:
        """No-op: per-prompt outlier detection requires no training corpus."""

    def score(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Score all samples. Processes in batches of 100 to limit memory."""
        scores: list[float] = []

        for batch_start in range(0, len(samples), _BATCH_SIZE):
            batch = samples[batch_start : batch_start + _BATCH_SIZE]
            for sample in batch:
                scores.append(self._score_one(sample))

        return np.array(scores, dtype=float)

    def score_single(self, sample: BenchmarkSample) -> float:
        """Score a single sample (used for latency profiling)."""
        return self._score_one(sample)

    # ------------------------------------------------------------------
    # Internal per-sample scoring
    # ------------------------------------------------------------------

    def _score_one(self, sample: BenchmarkSample) -> float:
        """Chunk, embed, fit outlier model, aggregate — for one prompt."""
        chunks = self._chunker.chunk(sample.prompt_body)

        # --- Fallback padding to reach min_chunks ---
        if len(chunks) < self.min_chunks:
            # Try splitting the longest chunk further with the fallback chunker
            chunks = _pad_chunks(chunks, self._fallback_chunker, self.min_chunks)

        if len(chunks) < self.min_chunks:
            # Still not enough material; outlier detection is degenerate
            logger.debug(
                "Sample %s has only %d chunks (< min_chunks=%d); returning 0.0",
                sample.id,
                len(chunks),
                self.min_chunks,
            )
            self._update_avg_chunks(1)
            return 0.0

        self._update_avg_chunks(len(chunks))

        # Embed all chunks at once (one batch call per prompt)
        embeddings = self.embedder.encode(chunks)  # (n_chunks, dim)

        # Fit outlier detector on THIS prompt's chunk embeddings
        anomaly_scores = self._fit_and_score_outlier(embeddings)

        return float(self._aggregate(anomaly_scores, embeddings))

    def _fit_and_score_outlier(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit a fresh outlier detector on chunk embeddings, return per-chunk scores."""
        n = len(embeddings)

        if self.outlier_detector_name == "lof":
            from sklearn.neighbors import LocalOutlierFactor

            n_neighbors = min(3, n - 1)
            detector = LocalOutlierFactor(n_neighbors=max(1, n_neighbors))
            detector.fit_predict(embeddings)
            return (-detector.negative_outlier_factor_).astype(float)
        elif self.outlier_detector_name == "iforest":
            from sklearn.ensemble import IsolationForest

            detector = IsolationForest(
                n_estimators=50,
                contamination=0.1,
                random_state=42,
            )
            detector.fit(embeddings)
            return (-detector.score_samples(embeddings)).astype(float)
        elif self.outlier_detector_name == "ecod":
            return _ecod_scores(embeddings)
        else:
            raise ValueError(f"Unknown outlier_detector: {self.outlier_detector_name!r}")

    def _aggregate(self, anomaly_scores: np.ndarray, embeddings: np.ndarray) -> float:
        """Aggregate per-chunk anomaly scores into a single prompt score."""
        if self.aggregation == "max":
            return float(np.max(anomaly_scores))

        elif self.aggregation == "top2_mean":
            k = min(2, len(anomaly_scores))
            top_k = np.sort(anomaly_scores)[-k:]
            return float(np.mean(top_k))

        elif self.aggregation == "pct_above_threshold":
            # Fraction of chunks above the 90th-percentile anomaly score
            threshold = float(np.percentile(anomaly_scores, 90))
            return float(np.mean(anomaly_scores > threshold))

        elif self.aggregation == "zscore_max":
            # Z-score of the maximum distance from the centroid
            centroid = embeddings.mean(axis=0)  # (dim,)
            distances = np.linalg.norm(embeddings - centroid, axis=1)  # (n_chunks,)
            if distances.std() < 1e-10:
                return 0.0
            z_scores = (distances - distances.mean()) / distances.std()
            return float(np.max(z_scores))

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

    def _update_avg_chunks(self, n_chunks: int) -> None:
        self._scored_count += 1
        # Exponential moving average (alpha = 0.05 so early values stabilise quickly)
        alpha = 0.05
        self._avg_chunks = (1 - alpha) * self._avg_chunks + alpha * n_chunks


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _pad_chunks(
    chunks: list[str],
    fallback_chunker: object,
    min_chunks: int,
) -> list[str]:
    """Try to reach at least min_chunks by re-splitting the longest chunk."""
    if len(chunks) >= min_chunks:
        return chunks

    # Sort by length descending and try to split the longest
    sorted_chunks = sorted(chunks, key=len, reverse=True)
    result = list(chunks)

    for candidate in sorted_chunks:
        sub = fallback_chunker.chunk(candidate)
        if len(sub) > 1:
            result.remove(candidate)
            result.extend(sub)
        if len(result) >= min_chunks:
            break

    return result


def _ecod_scores(embeddings: np.ndarray) -> np.ndarray:
    """Compute ECOD-style empirical tail scores without an external dependency."""
    n, _ = embeddings.shape
    if n == 0:
        return np.array([], dtype=float)

    ranks_low = _column_ranks(embeddings)
    ranks_high = n - ranks_low + 1

    tail_low = -np.log(ranks_low / (n + 1.0))
    tail_high = -np.log(ranks_high / (n + 1.0))
    return np.max(np.maximum(tail_low, tail_high), axis=1).astype(float)


def _column_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ascending ranks per column."""
    order = np.argsort(values, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(1, values.shape[0] + 1, dtype=float)[:, None]
    np.put_along_axis(ranks, order, row_indices, axis=0)
    return ranks
