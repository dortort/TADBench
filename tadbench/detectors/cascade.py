"""Cascade pipeline: TPS -> IPSOD -> ICCD with early exit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.base import BaseDetector
from tadbench.detectors.intent_content import IntentContentDetector
from tadbench.detectors.intra_prompt_outlier import IntraPromptOutlierDetector
from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
from tadbench.embedders.protocol import EmbedderProtocol

logger = logging.getLogger(__name__)


@dataclass
class CascadeMetadata:
    """Per-sample metadata produced by the cascade pipeline."""

    stage_reached: int  # 1, 2, or 3
    stage1_score: float
    stage2_score: Optional[float] = None
    stage3_score: Optional[float] = None
    final_score: float = 0.0


@dataclass
class CascadeAggregateMetrics:
    """Aggregate efficiency metrics across all cascade-scored samples."""

    avg_stage_reached: float
    pct_filtered_stage1: float   # fraction classified at stage 1
    pct_filtered_stage2: float   # fraction classified at stages 1 OR 2
    combined_latency_ms: float   # weighted sum of per-stage latencies


class CascadeDetector(BaseDetector):
    """Three-stage cascade detection pipeline.

    Stage 1 (cheap):  ThreatPrototypeDetector — FAISS similarity search.
    Stage 2 (medium): IntraPromptOutlierDetector — per-chunk outlier scoring.
    Stage 3 (expensive): IntentContentDetector — description vs body mismatch.

    Early exit: samples exceeding stage1_threshold at stage 1 are immediately
    labelled malicious (skipping stages 2 & 3). Samples passing stage 1 but
    exceeding stage2_threshold at stage 2 skip stage 3. Only samples passing
    both gates reach the full ICCD stage.

    This mirrors a deployment scenario where cheap filters guard expensive ones.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        stage1_threshold: float = 0.8,
        stage2_threshold: float = 0.7,
        stage3_threshold: float = 0.5,
        tps_kwargs: Optional[dict] = None,
        ipsod_kwargs: Optional[dict] = None,
        iccd_kwargs: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(embedder, **kwargs)
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.stage3_threshold = stage3_threshold

        tps_kwargs = tps_kwargs or {}
        ipsod_kwargs = ipsod_kwargs or {}
        iccd_kwargs = iccd_kwargs or {}

        self._stage1 = ThreatPrototypeDetector(embedder, **tps_kwargs)
        self._stage2 = IntraPromptOutlierDetector(embedder, **ipsod_kwargs)
        self._stage3 = IntentContentDetector(embedder, **iccd_kwargs)

        # Latency accumulators (seconds, updated during score_with_metadata calls)
        self._stage_latencies: list[float] = [0.0, 0.0, 0.0]
        self._stage_call_counts: list[int] = [0, 0, 0]

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            f"cascade_s1_{self.stage1_threshold}_s2_{self.stage2_threshold}"
            f"_s3_{self.stage3_threshold}"
        )

    @property
    def embedding_calls_per_sample(self) -> int:
        # Weighted average based on typical filtering rates (rough estimate)
        return 1

    def fit(self, samples: list[BenchmarkSample]) -> None:
        """Delegate fit() to all three sub-detectors."""
        logger.info("Fitting stage 1 (ThreatPrototype)...")
        self._stage1.fit(samples)
        logger.info("Fitting stage 2 (IntraPromptOutlier)...")
        self._stage2.fit(samples)
        logger.info("Fitting stage 3 (IntentContent)...")
        self._stage3.fit(samples)

    def score(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Return binary pipeline decisions (1 = malicious), not mixed raw scores."""
        scores, _ = self.score_with_metadata(samples)
        return scores

    def score_single(self, sample: BenchmarkSample) -> float:
        """Score a single sample through the cascade."""
        scores, _ = self.score_with_metadata([sample])
        return float(scores[0])

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def score_with_metadata(
        self, samples: list[BenchmarkSample]
    ) -> tuple[np.ndarray, list[CascadeMetadata]]:
        """Run cascade, returning both final scores and per-sample metadata.

        Returns
        -------
        scores : np.ndarray, shape (n,)
        metadata : list[CascadeMetadata], length n
        """
        self._stage_latencies = [0.0, 0.0, 0.0]
        self._stage_call_counts = [0, 0, 0]
        n = len(samples)
        if not n:
            return np.array([], dtype=float), []
        final_scores = np.zeros(n, dtype=float)
        metadata: list[CascadeMetadata] = []

        # --- Stage 1: ThreatPrototype ---
        t0 = time.perf_counter()
        s1_scores = self._stage1.score(samples)
        stage1_elapsed = time.perf_counter() - t0
        self._record_latency(0, stage1_elapsed, n)

        # Partition samples into early-exit vs continue-to-stage-2
        stage1_exit_mask = s1_scores >= self.stage1_threshold
        stage2_indices = [i for i in range(n) if not stage1_exit_mask[i]]

        # Record stage-1 exits
        for i in range(n):
            if stage1_exit_mask[i]:
                final_scores[i] = 1.0
                metadata.append(
                    CascadeMetadata(
                        stage_reached=1,
                        stage1_score=float(s1_scores[i]),
                        final_score=1.0,
                    )
                )

        if not stage2_indices:
            meta_list = _build_ordered_metadata(
                n, stage1_exit_mask, s1_scores, {}, {}
            )
            return final_scores, meta_list

        # --- Stage 2: IntraPromptOutlier ---
        stage2_samples = [samples[i] for i in stage2_indices]
        t0 = time.perf_counter()
        s2_scores = self._stage2.score(stage2_samples)
        stage2_elapsed = time.perf_counter() - t0
        self._record_latency(1, stage2_elapsed, len(stage2_samples))

        stage2_exit_mask = s2_scores >= self.stage2_threshold
        stage3_local_indices = [j for j, exit_ in enumerate(stage2_exit_mask) if not exit_]
        stage3_global_indices = [stage2_indices[j] for j in stage3_local_indices]

        # Record stage-2 exits
        stage2_exit_meta: dict[int, CascadeMetadata] = {}
        for j, global_i in enumerate(stage2_indices):
            if stage2_exit_mask[j]:
                final_scores[global_i] = 1.0
                stage2_exit_meta[global_i] = CascadeMetadata(
                    stage_reached=2,
                    stage1_score=float(s1_scores[global_i]),
                    stage2_score=float(s2_scores[j]),
                    final_score=1.0,
                )

        if not stage3_global_indices:
            meta_list = _build_ordered_metadata(
                n, stage1_exit_mask, s1_scores, stage2_exit_meta, {}
            )
            return final_scores, meta_list

        # --- Stage 3: IntentContent ---
        stage3_samples = [samples[i] for i in stage3_global_indices]
        t0 = time.perf_counter()
        s3_scores = self._stage3.score(stage3_samples)
        stage3_elapsed = time.perf_counter() - t0
        self._record_latency(2, stage3_elapsed, len(stage3_samples))

        # Build O(1) lookup from global index to stage-2 position.
        _s2_pos = {gi: pos for pos, gi in enumerate(stage2_indices)}

        stage3_exit_meta: dict[int, CascadeMetadata] = {}
        for k, global_i in enumerate(stage3_global_indices):
            s3 = float(s3_scores[k])
            # Stage evidence stays in metadata; final scores are decisions.
            fallback_s2 = float(s2_scores[_s2_pos[global_i]])
            if not np.isfinite(s3):
                raise ValueError("Cascade stage 3 requires a finite score for every sample")
            actual_final = float(s3 >= self.stage3_threshold)
            final_scores[global_i] = actual_final
            stage3_exit_meta[global_i] = CascadeMetadata(
                stage_reached=3,
                stage1_score=float(s1_scores[global_i]),
                stage2_score=fallback_s2,
                stage3_score=None if np.isnan(s3) else s3,
                final_score=actual_final,
            )

        meta_list = _build_ordered_metadata(
            n, stage1_exit_mask, s1_scores, stage2_exit_meta, stage3_exit_meta
        )
        return final_scores, meta_list

    def aggregate_metrics(self, metadata: list[CascadeMetadata]) -> CascadeAggregateMetrics:
        """Compute aggregate cascade metrics from a list of per-sample metadata."""
        if not metadata:
            return CascadeAggregateMetrics(
                avg_stage_reached=0.0,
                pct_filtered_stage1=0.0,
                pct_filtered_stage2=0.0,
                combined_latency_ms=0.0,
            )

        n = len(metadata)
        stages = [m.stage_reached for m in metadata]
        avg_stage = float(np.mean(stages))
        pct_s1 = sum(1 for s in stages if s == 1) / n
        pct_s1_or_s2 = sum(1 for s in stages if s <= 2) / n

        # Total stage work divided by original traffic includes reach fractions.
        combined_latency = sum(self._stage_latencies) * 1000 / n

        return CascadeAggregateMetrics(
            avg_stage_reached=avg_stage,
            pct_filtered_stage1=pct_s1,
            pct_filtered_stage2=pct_s1_or_s2,
            combined_latency_ms=combined_latency,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_latency(self, stage: int, elapsed_s: float, n_samples: int) -> None:
        if n_samples > 0:
            self._stage_latencies[stage] += elapsed_s
            self._stage_call_counts[stage] += n_samples


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_ordered_metadata(
    n: int,
    stage1_exit_mask: np.ndarray,
    s1_scores: np.ndarray,
    stage2_exit_meta: dict[int, CascadeMetadata],
    stage3_exit_meta: dict[int, CascadeMetadata],
) -> list[CascadeMetadata]:
    """Reconstruct the metadata list in original sample order."""
    result: list[CascadeMetadata] = []
    for i in range(n):
        if stage1_exit_mask[i]:
            result.append(
                CascadeMetadata(
                    stage_reached=1,
                    stage1_score=float(s1_scores[i]),
                    final_score=1.0,
                )
            )
        elif i in stage2_exit_meta:
            result.append(stage2_exit_meta[i])
        elif i in stage3_exit_meta:
            result.append(stage3_exit_meta[i])
        else:
            # Should not happen; safety fallback
            result.append(
                CascadeMetadata(
                    stage_reached=1,
                    stage1_score=float(s1_scores[i]),
                    final_score=1.0,
                )
            )
    return result

