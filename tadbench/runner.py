"""TADBench benchmark harness: orchestrates embed -> detect -> evaluate pipeline.

This module is the primary library API entry point.  Import and use directly
without going through the CLI::

    from tadbench.runner import BenchmarkRunner

    runner = BenchmarkRunner(embedder_tier="small", cache=True)
    result = runner.run_single(
        technique="threat_prototype",
        dataset_path="data/processed/benchmark_test.jsonl",
        threat_bank_path="data/processed/threat_bank.jsonl",
    )
    print(result.auroc)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tadbench.config import (
    CACHE_DIR,
    DEFAULT_DETECTOR_CONFIG,
    RESULTS_DIR,
    TECHNIQUE_DATASET_COMPAT,
)
from tadbench.compat import validate_dataset_for_technique
from tadbench.data.schema import BenchmarkSample, validate_sample
from tadbench.detectors.base import BaseDetector
from tadbench.detectors.cascade import CascadeDetector
from tadbench.detectors.intent_content import IntentContentDetector
from tadbench.detectors.intra_prompt_outlier import IntraPromptOutlierDetector
from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
from tadbench.embedders.cache import EmbeddingCache
from tadbench.embedders.protocol import EmbedderProtocol
from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
from tadbench.evaluation.calibration import calibrate_threshold
from tadbench.evaluation.drift import DriftEvaluator, DriftReport
from tadbench.evaluation.efficiency import (
    EfficiencyReport,
    LatencyProfiler,
    measure_scaling_curves,
    resolve_matrix_profiling_mode,
)
from tadbench.evaluation.metrics import compute_all_metrics
from tadbench.utils.io import stream_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """All detection quality and efficiency metrics for a single detector run.

    Produced by :meth:`BenchmarkRunner.run_single`.  Serialisable to JSON via
    :meth:`to_dict` and to a flat CSV row via :meth:`to_csv_row`.
    """

    # Identity
    detector_name: str
    embedder: str
    dataset_slice: str

    # Detection quality
    auroc: float
    auprc: float
    tpr_at_fpr: dict[float, float]           # {fpr_target -> tpr}
    calibration_error: float

    # Breakdowns
    per_attack_recall: dict[str, float]      # {attack_type -> recall}
    per_group_metrics: dict[str, dict]       # full per_group output from compute_all_metrics

    # Efficiency
    efficiency: EfficiencyReport

    # Cascade-specific (None for non-cascade detectors)
    cascade_metrics: Optional[dict] = None
    # keys: avg_stage_reached, pct_filtered_stage1, pct_filtered_stage2, combined_latency_ms

    # Calibration metadata
    target_fpr: float = 0.005
    calibrated_threshold: Optional[float] = None
    benchmark_mode: str = "full"

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON output."""
        eff = self.efficiency
        return {
            "detector_name": self.detector_name,
            "embedder": self.embedder,
            "dataset_slice": self.dataset_slice,
            "target_fpr": self.target_fpr,
            "calibrated_threshold": self.calibrated_threshold,
            "benchmark_mode": self.benchmark_mode,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "tpr_at_fpr": {str(k): v for k, v in self.tpr_at_fpr.items()},
            "calibration_error": self.calibration_error,
            "per_attack_recall": self.per_attack_recall,
            "per_group_metrics": self.per_group_metrics,
            "efficiency": {
                "p50_ms": eff.p50_ms,
                "p95_ms": eff.p95_ms,
                "p99_ms": eff.p99_ms,
                "throughput_rps": eff.throughput_rps,
                "peak_rss_mb": eff.peak_rss_mb,
                "cpu_utilization_pct": eff.cpu_utilization_pct,
                "index_build_time_s": eff.index_build_time_s,
                "index_update_cost_s": eff.index_update_cost_s,
                "embedding_calls_per_sample": eff.embedding_calls_per_sample,
                "total_tokens_embedded": eff.total_tokens_embedded,
                "cross_encoder_calls_per_sample": eff.cross_encoder_calls_per_sample,
                "latency_profile_samples": eff.latency_profile_samples,
                "throughput_profile_samples": eff.throughput_profile_samples,
            },
            "cascade_metrics": self.cascade_metrics,
        }

    def to_csv_row(self) -> dict:
        """Return a flat dict suitable for constructing a pandas DataFrame row.

        Columns match acceptance criterion 11:
        detector_name, embedder, dataset_slice, attack_type, fpr_target, tpr,
        auroc, auprc, calibration_error, p50_ms, p95_ms, p99_ms, rss_mb,
        throughput_rps, embedding_calls, total_tokens, cross_encoder_calls.

        One row per (detector, dataset_slice, fpr_target) combination.
        The attack_type column contains "all" for the aggregate row.
        """
        eff = self.efficiency
        base: dict = {
            "detector_name": self.detector_name,
            "embedder": self.embedder,
            "dataset_slice": self.dataset_slice,
            "benchmark_mode": self.benchmark_mode,
            "attack_type": "all",
            "fpr_target": self.target_fpr,
            "tpr": self.tpr_at_fpr.get(self.target_fpr, float("nan")),
            "auroc": self.auroc,
            "auprc": self.auprc,
            "calibration_error": self.calibration_error,
            "p50_ms": eff.p50_ms,
            "p95_ms": eff.p95_ms,
            "p99_ms": eff.p99_ms,
            "rss_mb": eff.peak_rss_mb,
            "throughput_rps": eff.throughput_rps,
            "embedding_calls": eff.embedding_calls_per_sample,
            "total_tokens": eff.total_tokens_embedded,
            "cross_encoder_calls": eff.cross_encoder_calls_per_sample,
            "latency_profile_samples": eff.latency_profile_samples,
            "throughput_profile_samples": eff.throughput_profile_samples,
        }
        # Add cascade fields (NaN when not applicable)
        if self.cascade_metrics:
            base["avg_stage_reached"] = self.cascade_metrics.get("avg_stage_reached", float("nan"))
            base["pct_filtered_stage1"] = self.cascade_metrics.get("pct_filtered_stage1", float("nan"))
            base["pct_filtered_stage2"] = self.cascade_metrics.get("pct_filtered_stage2", float("nan"))
            base["combined_latency_ms"] = self.cascade_metrics.get("combined_latency_ms", float("nan"))
        else:
            base["avg_stage_reached"] = float("nan")
            base["pct_filtered_stage1"] = float("nan")
            base["pct_filtered_stage2"] = float("nan")
            base["combined_latency_ms"] = float("nan")
        return base


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrate the full TADBench pipeline for one or many technique runs.

    Parameters
    ----------
    embedder_tier:
        Embedding model tier: ``"small"``, ``"medium"``, or ``"large"``.
        Alternatively a full HuggingFace model identifier.
    cache:
        Whether to wrap the embedder in :class:`~tadbench.embedders.cache.EmbeddingCache`
        to avoid redundant computation during matrix runs.
    results_dir:
        Directory where CSV/JSON result files are written.
    device:
        PyTorch device string for the embedder (``"cpu"``, ``"cuda"``, …).
    """

    def __init__(
        self,
        embedder_tier: str,
        cache: bool = True,
        results_dir: str | Path = RESULTS_DIR,
        device: str = "cpu",
    ) -> None:
        self.embedder_tier = embedder_tier
        self.cache = cache
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Build embedder (and optionally wrap in cache)
        base_embedder: EmbedderProtocol = SentenceTransformerEmbedder(
            embedder_tier, device=device
        )
        if cache:
            self._embedder: EmbedderProtocol = EmbeddingCache(
                base_embedder, cache_dir=str(CACHE_DIR)
            )
        else:
            self._embedder = base_embedder

        logger.info(
            "BenchmarkRunner initialised: tier=%s cache=%s results_dir=%s",
            embedder_tier,
            cache,
            self.results_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        technique: str,
        dataset_path: str | Path,
        detector_config: Optional[dict] = None,
        target_fpr: float = 0.005,
        val_dataset_path: Optional[str | Path] = None,
        threat_bank_path: Optional[str | Path] = None,
        benchmark_mode: str = "full",
    ) -> BenchmarkResult:
        """Run a single detection technique on a dataset.

        Parameters
        ----------
        technique:
            One of ``"threat_prototype"``, ``"intra_prompt_outlier"``,
            ``"intent_content_consistency"``, or ``"cascade"``.
        dataset_path:
            Path to the test JSONL file.
        detector_config:
            Optional dict of detector hyperparameter overrides merged on top of
            :data:`~tadbench.config.DEFAULT_DETECTOR_CONFIG`.
        target_fpr:
            FPR target for TPR@FPR and threshold calibration.
        val_dataset_path:
            If provided, calibrate the detection threshold using this validation
            JSONL file instead of deriving it from the test set.
        threat_bank_path:
            Path to a threat-bank JSONL file.  Required for ``threat_prototype``
            and ``cascade`` techniques.
        benchmark_mode:
            Efficiency profiling mode: ``"full"`` preserves the exhaustive
            matrix profiling behavior, while ``"fast"`` uses sampled estimates.

        Returns
        -------
        BenchmarkResult
            All detection quality and efficiency metrics.

        Raises
        ------
        ValueError
            If the technique is not compatible with the dataset (checked via
            ``TECHNIQUE_DATASET_COMPAT``), or if required arguments are missing.
        """
        dataset_path = Path(dataset_path)
        if benchmark_mode not in {"full", "fast"}:
            raise ValueError("benchmark_mode must be 'full' or 'fast'")
        _validate_technique(technique)
        _check_dataset_compat(technique, dataset_path)

        # Load samples
        test_samples = self._load_samples(dataset_path)
        if not test_samples:
            raise ValueError(f"Dataset is empty: {dataset_path}")
        validate_dataset_for_technique(technique, test_samples, dataset_path)

        # Build detector
        merged_config = _merge_config(technique, detector_config)
        detector = self._create_detector(technique, merged_config)

        # Fit detector
        self._fit_detector(detector, technique, test_samples, threat_bank_path)

        # Score test set (cascade path is handled specially)
        y_true = np.array(
            [1 if s.label == "malicious" else 0 for s in test_samples], dtype=int
        )
        attack_groups = [s.attack_type for s in test_samples]

        cascade_metrics: Optional[dict] = None

        if technique == "cascade" and isinstance(detector, CascadeDetector):
            y_score, cascade_meta_list = detector.score_with_metadata(test_samples)
            agg = detector.aggregate_metrics(cascade_meta_list)
            cascade_metrics = {
                "avg_stage_reached": agg.avg_stage_reached,
                "pct_filtered_stage1": agg.pct_filtered_stage1,
                "pct_filtered_stage2": agg.pct_filtered_stage2,
                "combined_latency_ms": agg.combined_latency_ms,
            }
        else:
            y_score = detector.score(test_samples)

        # Optional threshold calibration from validation set
        calibrated_threshold: Optional[float] = None
        if val_dataset_path is not None:
            val_samples = self._load_samples(Path(val_dataset_path))
            if val_samples:
                validate_dataset_for_technique(technique, val_samples, Path(val_dataset_path))
                self._fit_detector(detector, technique, val_samples, threat_bank_path)
                val_scores = detector.score(val_samples)
                val_labels = np.array(
                    [1 if s.label == "malicious" else 0 for s in val_samples], dtype=int
                )
                # Restore the original fit after calibration scoring
                self._fit_detector(detector, technique, test_samples, threat_bank_path)
                try:
                    calibrated_threshold = calibrate_threshold(
                        val_labels, val_scores, target_fpr
                    )
                except ValueError as exc:
                    logger.warning("Threshold calibration failed: %s", exc)

        # Compute detection quality metrics
        metrics = compute_all_metrics(
            y_true,
            y_score,
            groups=attack_groups,
            target_fprs=[0.001, 0.005, 0.01],
        )

        # Extract per-attack recall at target_fpr
        per_attack_recall = _extract_per_attack_recall(
            metrics.get("per_group", {}), target_fpr
        )

        # Profile efficiency
        profiling_cfg = resolve_matrix_profiling_mode(benchmark_mode)
        profiler = LatencyProfiler()
        efficiency = profiler.profile_detector(
            detector,
            test_samples,
            n_runs=profiling_cfg.latency_runs,
            max_latency_samples=profiling_cfg.max_latency_samples,
            max_throughput_samples=profiling_cfg.max_throughput_samples,
        )

        return BenchmarkResult(
            detector_name=detector.name,
            embedder=self.embedder_tier,
            dataset_slice=dataset_path.name,
            auroc=metrics["auroc"],
            auprc=metrics["auprc"],
            tpr_at_fpr=metrics["tpr_at_fpr"],
            calibration_error=metrics["calibration_error"],
            per_attack_recall=per_attack_recall,
            per_group_metrics=metrics.get("per_group", {}),
            efficiency=efficiency,
            cascade_metrics=cascade_metrics,
            target_fpr=target_fpr,
            calibrated_threshold=calibrated_threshold,
            benchmark_mode=benchmark_mode,
        )

    def run_matrix(self, config_path: str | Path, benchmark_mode: str = "full") -> pd.DataFrame:
        """Run the full benchmark matrix defined by a JSON config file.

        The config schema matches ``matrix_config.json`` in the project root.
        Each (embedder, technique variant, dataset) combination produces one
        :class:`BenchmarkResult`; all results are collected into a DataFrame
        and saved as CSV to ``results_dir/matrix_results.csv``.

        Parameters
        ----------
        config_path:
            Path to the matrix configuration JSON file.
        benchmark_mode:
            Efficiency profiling mode forwarded to :meth:`run_single`.

        Returns
        -------
        pd.DataFrame
            One row per completed run, with columns from
            :meth:`BenchmarkResult.to_csv_row`.
        """
        config_path = Path(config_path)
        if benchmark_mode not in {"full", "fast"}:
            raise ValueError("benchmark_mode must be 'full' or 'fast'")
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        rows: list[dict] = []
        embedders: list[str] = config.get("embedders", [self.embedder_tier])
        target_fprs: list[float] = config.get("target_fprs", [0.001, 0.005, 0.01])
        n_efficiency_runs: int = config.get("n_efficiency_runs", 3)

        datasets: list[str] = config.get("datasets", [])
        techniques_cfg: list[dict] = config.get("techniques", [])

        total = sum(
            len(embedders) * len(t.get("variants", [{}])) * len(datasets)
            for t in techniques_cfg
        )
        completed = 0

        for emb_tier in embedders:
            runner = self if emb_tier == self.embedder_tier else BenchmarkRunner(
                embedder_tier=emb_tier,
                cache=self.cache,
                results_dir=self.results_dir,
                device=self.device,
            )
            for tech_entry in techniques_cfg:
                technique: str = tech_entry["name"]
                variants: list[dict] = tech_entry.get("variants", [{}])

                for variant in variants:
                    for dataset_path in datasets:
                        completed += 1
                        logger.info(
                            "[%d/%d] embedder=%s technique=%s variant=%s dataset=%s",
                            completed,
                            total,
                            emb_tier,
                            technique,
                            variant,
                            dataset_path,
                        )
                        try:
                            result = runner.run_single(
                                technique=technique,
                                dataset_path=dataset_path,
                                detector_config=variant,
                                target_fpr=target_fprs[0] if target_fprs else 0.005,
                                benchmark_mode=benchmark_mode,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "run_matrix: run failed for embedder=%s technique=%s variant=%s dataset=%s: %s",
                                emb_tier,
                                technique,
                                variant,
                                dataset_path,
                                exc,
                            )
                            continue

                        # Expand result to one row per fpr_target
                        for fpr in target_fprs:
                            row = result.to_csv_row()
                            row["fpr_target"] = fpr
                            row["tpr"] = result.tpr_at_fpr.get(fpr, float("nan"))
                            rows.append(row)

                        # Per-attack-type breakdown rows
                        if config.get("attack_type_breakdown", True):
                            for attack_type, group_data in result.per_group_metrics.items():
                                for fpr in target_fprs:
                                    group_row = result.to_csv_row()
                                    group_row["attack_type"] = attack_type
                                    group_row["fpr_target"] = fpr
                                    group_row["tpr"] = group_data.get(
                                        "tpr_at_fpr", {}
                                    ).get(fpr, float("nan"))
                                    group_row["auroc"] = group_data.get("auroc", float("nan"))
                                    group_row["auprc"] = group_data.get("auprc", float("nan"))
                                    rows.append(group_row)

        df = pd.DataFrame(rows)
        out_path = self.results_dir / "matrix_results.csv"
        df.to_csv(out_path, index=False)
        logger.info("run_matrix: wrote %d rows to %s", len(df), out_path)
        return df

    def run_drift_eval(
        self,
        holdout_attack: str,
        dataset_path: str | Path,
        technique: str,
        detector_config: Optional[dict] = None,
        target_fpr: float = 0.005,
    ) -> DriftReport:
        """Evaluate how detection degrades for a held-out attack family.

        Parameters
        ----------
        holdout_attack:
            The ``attack_type`` value to exclude from training/validation (e.g.
            ``"jailbreak"``).
        dataset_path:
            Path to the full dataset JSONL (all splits merged).
        technique:
            Detection technique to evaluate.
        detector_config:
            Optional hyperparameter overrides.
        target_fpr:
            FPR used for threshold calibration in the drift protocol.

        Returns
        -------
        DriftReport
            Recall degradation, threshold drift, and retraining cost.
        """
        dataset_path = Path(dataset_path)
        _validate_technique(technique)

        all_samples = self._load_samples(dataset_path)
        validate_dataset_for_technique(technique, all_samples, dataset_path)

        evaluator = DriftEvaluator(holdout_attack_type=holdout_attack)
        train, val, test = evaluator.create_drift_splits(all_samples)

        merged_config = _merge_config(technique, detector_config)
        detector = self._create_detector(technique, merged_config)

        report = evaluator.evaluate_drift(
            detector=detector,
            train=train,
            val=val,
            test=test,
            target_fpr=target_fpr,
        )
        return report

    def run_scaling(
        self,
        technique: str,
        dataset_path: str | Path,
        detector_config: Optional[dict] = None,
        threat_bank_path: Optional[str | Path] = None,
    ) -> dict:
        """Measure latency scaling curves for a detector.

        Runs ``measure_scaling_curves`` for each variable that is relevant to
        the given technique:

        - ``threat_prototype``: prompt_length, threat_bank_size
        - ``intra_prompt_outlier``: prompt_length, chunks_per_prompt
        - ``intent_content_consistency``: prompt_length, pair_length
        - ``cascade``: prompt_length

        Parameters
        ----------
        technique:
            Detection technique to profile.
        dataset_path:
            Path to the test JSONL file.
        detector_config:
            Optional hyperparameter overrides.
        threat_bank_path:
            Path to the threat bank JSONL (required for threat_prototype /
            cascade scaling).

        Returns
        -------
        dict
            Mapping of variable name -> list of per-bin measurement dicts.
            Each dict has keys: variable, variable_value, n_samples, p50_ms, p95_ms.
        """
        dataset_path = Path(dataset_path)
        _validate_technique(technique)

        samples = self._load_samples(dataset_path)
        validate_dataset_for_technique(technique, samples, dataset_path)
        validate_dataset_for_technique(technique, samples, dataset_path)

        merged_config = _merge_config(technique, detector_config)
        detector = self._create_detector(technique, merged_config)

        self._fit_detector(detector, technique, samples, threat_bank_path)

        # Define scaling variables per technique
        scaling_variables: dict[str, tuple[str, list]] = _scaling_variables_for(technique)

        results: dict[str, list[dict]] = {}
        for var_name, (variable_key, values) in scaling_variables.items():
            logger.info("run_scaling: measuring variable='%s'...", variable_key)
            try:
                curve = measure_scaling_curves(
                    detector=detector,
                    samples=samples,
                    variable=variable_key,
                    values=values,
                )
                results[var_name] = curve
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_scaling: '%s' failed: %s", variable_key, exc)
                results[var_name] = []

        return results

    def save_results(
        self,
        results: list[BenchmarkResult],
        output_path: str | Path,
    ) -> None:
        """Save a list of BenchmarkResult objects as a CSV file.

        Columns match acceptance criterion 11 (detector_name, embedder,
        dataset_slice, attack_type, fpr_target, tpr, auroc, auprc,
        calibration_error, p50_ms, p95_ms, p99_ms, rss_mb, throughput_rps,
        embedding_calls, total_tokens, cross_encoder_calls).

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` instances to serialise.
        output_path:
            Destination CSV file path.  Parent directories are created if
            they do not exist.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for result in results:
            # Aggregate row (attack_type="all")
            rows.append(result.to_csv_row())
            # Per-attack-type breakdown rows
            for attack_type, group_data in result.per_group_metrics.items():
                row = result.to_csv_row()
                row["attack_type"] = attack_type
                row["tpr"] = group_data.get("tpr_at_fpr", {}).get(
                    result.target_fpr, float("nan")
                )
                row["auroc"] = group_data.get("auroc", float("nan"))
                row["auprc"] = group_data.get("auprc", float("nan"))
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info("save_results: wrote %d rows to %s", len(df), output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_detector(
        self,
        technique: str,
        detector_config: Optional[dict] = None,
    ) -> BaseDetector:
        """Factory that maps technique names to detector classes.

        Parameters
        ----------
        technique:
            Technique name (see ``TECHNIQUE_DATASET_COMPAT`` keys).
        detector_config:
            Hyperparameter dict already merged with defaults.  Forwarded as
            keyword arguments to the detector ``__init__``.

        Returns
        -------
        BaseDetector
            Un-fitted detector instance.

        Raises
        ------
        ValueError
            If ``technique`` is not a known technique name.
        """
        cfg = detector_config or {}

        if technique == "threat_prototype":
            return ThreatPrototypeDetector(self._embedder, **cfg)
        elif technique == "intra_prompt_outlier":
            return IntraPromptOutlierDetector(self._embedder, **cfg)
        elif technique == "intent_content_consistency":
            return IntentContentDetector(self._embedder, **cfg)
        elif technique == "cascade":
            # CascadeDetector accepts stage1_threshold, stage2_threshold plus
            # optional tps_kwargs / ipsod_kwargs / iccd_kwargs.
            return CascadeDetector(self._embedder, **cfg)
        else:
            raise ValueError(
                f"Unknown technique '{technique}'. "
                f"Valid: {list(TECHNIQUE_DATASET_COMPAT.keys())}"
            )

    def _load_samples(self, dataset_path: Path) -> list[BenchmarkSample]:
        """Read and validate a JSONL dataset file.

        Invalid records are skipped with a warning; valid records are returned
        as :class:`~tadbench.data.schema.BenchmarkSample` instances.

        Parameters
        ----------
        dataset_path:
            Path to the JSONL file.

        Returns
        -------
        list[BenchmarkSample]
        """
        samples: list[BenchmarkSample] = []
        n_errors = 0
        for line_num, record in enumerate(stream_jsonl(dataset_path), start=1):
            try:
                sample = validate_sample(record)
                samples.append(sample)
            except ValueError as exc:
                n_errors += 1
                logger.warning(
                    "_load_samples: skipping line %d in %s: %s",
                    line_num,
                    dataset_path.name,
                    exc,
                )

        if n_errors:
            logger.warning(
                "_load_samples: %d invalid record(s) skipped out of %d total in %s",
                n_errors,
                len(samples) + n_errors,
                dataset_path.name,
            )

        logger.debug(
            "_load_samples: loaded %d samples from %s", len(samples), dataset_path.name
        )
        return samples

    def _fit_detector(
        self,
        detector: BaseDetector,
        technique: str,
        samples: list[BenchmarkSample],
        threat_bank_path: Optional[str | Path],
    ) -> None:
        """Fit a detector using benchmark-specific semantics for each technique."""
        if technique == "threat_prototype":
            fit_samples = self._resolve_fit_samples(technique, samples, threat_bank_path)
            logger.info("Fitting detector '%s' on %d samples...", detector.name, len(fit_samples))
            detector.fit(fit_samples)
            return

        if technique == "cascade" and isinstance(detector, CascadeDetector):
            stage1_fit_samples = self._resolve_fit_samples(
                "threat_prototype", samples, threat_bank_path
            )
            logger.info(
                "Fitting cascade detector: stage1=%d stage2/3=%d",
                len(stage1_fit_samples),
                len(samples),
            )
            detector._stage1.fit(stage1_fit_samples)
            detector._stage2.fit(samples)
            detector._stage3.fit(samples)
            return

        logger.info("Fitting detector '%s' on %d samples...", detector.name, len(samples))
        detector.fit(samples)

    def _resolve_fit_samples(
        self,
        technique: str,
        test_samples: list[BenchmarkSample],
        threat_bank_path: Optional[str | Path],
    ) -> list[BenchmarkSample]:
        """Determine what samples to pass to ``detector.fit()``.

        For ``threat_prototype`` and ``cascade``, the threat bank JSONL is
        loaded when provided; otherwise the malicious test samples are used as
        a fallback (useful for testing, not production).

        For ``intra_prompt_outlier`` and ``intent_content_consistency``,
        ``fit()`` is a no-op, so we simply return an empty list.

        Parameters
        ----------
        technique:
            Active technique name.
        test_samples:
            Loaded test-set samples (fallback when no threat bank provided).
        threat_bank_path:
            Optional path to threat-bank JSONL.

        Returns
        -------
        list[BenchmarkSample]
        """
        if technique in ("intra_prompt_outlier", "intent_content_consistency"):
            # fit() is a no-op for these detectors
            return test_samples  # still called; just harmless

        # threat_prototype / cascade — need a threat bank
        if threat_bank_path is not None:
            bank_path = Path(threat_bank_path)
            if bank_path.exists():
                bank = self._load_samples(bank_path)
                if bank:
                    logger.info(
                        "_resolve_fit_samples: loaded threat bank with %d samples from %s",
                        len(bank),
                        bank_path.name,
                    )
                    return bank
            logger.warning(
                "_resolve_fit_samples: threat_bank_path '%s' not found or empty; "
                "falling back to malicious test samples",
                threat_bank_path,
            )

        # Fallback: use malicious samples from test set
        malicious = [s for s in test_samples if s.label == "malicious"]
        if not malicious:
            logger.warning(
                "_resolve_fit_samples: no threat bank and no malicious test samples; "
                "fitting on all test samples as a last resort"
            )
            return test_samples

        logger.warning(
            "_resolve_fit_samples: no threat_bank_path provided; "
            "fitting on %d malicious test samples (not production-quality)",
            len(malicious),
        )
        return malicious


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _validate_technique(technique: str) -> None:
    """Raise ValueError if *technique* is not a known technique name."""
    known = set(TECHNIQUE_DATASET_COMPAT.keys())
    if technique not in known:
        raise ValueError(
            f"Unknown technique '{technique}'. Valid techniques: {sorted(known)}"
        )


def _check_dataset_compat(technique: str, dataset_path: Path) -> None:
    """Warn (or raise) if the dataset may be incompatible with the technique.

    Compatibility is inferred from the dataset filename stem against the
    :data:`~tadbench.config.TECHNIQUE_DATASET_COMPAT` mapping.  An unknown
    filename is allowed through with a warning (not all filenames encode the
    source name).
    """
    compat_sources = TECHNIQUE_DATASET_COMPAT.get(technique, [])
    stem = dataset_path.stem.lower()

    # Check if any compatible source name appears in the dataset file stem
    matched_any = any(src in stem for src in compat_sources)
    # Also allow generic names like "benchmark_combined", "test", "val", etc.
    generic_ok = any(
        keyword in stem
        for keyword in ("benchmark", "test", "val", "combined", "split")
    )

    if not matched_any and not generic_ok:
        logger.warning(
            "_check_dataset_compat: technique '%s' expects datasets from %s but "
            "dataset file '%s' does not match any compatible source name. "
            "Proceeding — set TECHNIQUE_DATASET_COMPAT to silence this warning.",
            technique,
            compat_sources,
            dataset_path.name,
        )


def _merge_config(technique: str, overrides: Optional[dict]) -> dict:
    """Merge DEFAULT_DETECTOR_CONFIG with caller-supplied overrides.

    Parameters
    ----------
    technique:
        Technique name used to look up defaults.
    overrides:
        Dict of hyperparameter overrides (may be None).

    Returns
    -------
    dict
        Merged hyperparameter dict.
    """
    defaults = dict(DEFAULT_DETECTOR_CONFIG.get(technique, {}))
    if overrides:
        defaults.update(overrides)
    return defaults


def _extract_per_attack_recall(
    per_group: dict[str, dict],
    target_fpr: float,
) -> dict[str, float]:
    """Extract TPR (recall) at *target_fpr* for each attack group.

    Parameters
    ----------
    per_group:
        Output of ``compute_per_group_metrics``, keyed by group label.
    target_fpr:
        The FPR level at which recall is reported.

    Returns
    -------
    dict[str, float]
        Mapping of attack_type -> TPR at *target_fpr*.
    """
    result: dict[str, float] = {}
    for group_name, group_data in per_group.items():
        tpr_map: dict = group_data.get("tpr_at_fpr", {})
        # tpr_at_fpr keys may be float or string depending on caller
        tpr_val = tpr_map.get(target_fpr, tpr_map.get(str(target_fpr), float("nan")))
        result[group_name] = float(tpr_val) if tpr_val is not None else float("nan")
    return result


def _scaling_variables_for(technique: str) -> dict[str, tuple[str, list]]:
    """Return the scaling variables to measure for a given technique.

    Returns
    -------
    dict mapping variable label -> (variable_key, values_list)
    """
    prompt_length_bins = [0, 128, 512, 2048, 99_999]
    threat_bank_sizes = [100, 500, 1_000, 5_000, 10_000]
    chunk_bins = [1, 3, 5, 10, 50]
    pair_length_bins = [0, 256, 512, 1_024, 4_096]

    if technique == "threat_prototype":
        return {
            "prompt_length": ("prompt_length", prompt_length_bins),
            "threat_bank_size": ("threat_bank_size", threat_bank_sizes),
        }
    elif technique == "intra_prompt_outlier":
        return {
            "prompt_length": ("prompt_length", prompt_length_bins),
            "chunks_per_prompt": ("chunks_per_prompt", chunk_bins),
        }
    elif technique == "intent_content_consistency":
        return {
            "prompt_length": ("prompt_length", prompt_length_bins),
            "pair_length": ("pair_length", pair_length_bins),
        }
    elif technique == "cascade":
        return {
            "prompt_length": ("prompt_length", prompt_length_bins),
        }
    else:
        return {
            "prompt_length": ("prompt_length", prompt_length_bins),
        }
