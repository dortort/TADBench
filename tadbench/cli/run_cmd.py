"""CLI command: bench.py run — run one or all detection techniques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from tadbench.compat import validate_dataset_for_technique
from tadbench.config import (
    DEFAULT_DETECTOR_CONFIG,
    PROCESSED_DIR,
    RESULTS_DIR,
)

ALL_TECHNIQUES = ["threat_prototype", "intra_prompt_outlier", "intent_content_consistency", "cascade"]

RUN_PROFILING_MODES: dict[str, dict[str, int | None]] = {
    "full": {
        "latency_runs": 1,
        "max_latency_samples": 200,
        "max_throughput_samples": None,
    },
    "fast": {
        "latency_runs": 1,
        "max_latency_samples": 50,
        "max_throughput_samples": 200,
    },
}


def _load_detector_config(detector_config: str | None) -> dict:
    """Parse detector_config as inline JSON string or JSON file path."""
    if detector_config is None:
        return {}
    # Try inline JSON first
    try:
        return json.loads(detector_config)
    except json.JSONDecodeError:
        pass
    # Try as file path
    path = Path(detector_config)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"Invalid JSON in detector config file {path}: {exc}"
            ) from exc
    raise click.ClickException(
        f"--detector-config value is neither valid JSON nor an existing file: {detector_config!r}"
    )


@click.command("run")
@click.option(
    "--technique",
    required=True,
    help='Detection technique or "all". One of: threat_prototype, intra_prompt_outlier, '
         "intent_content_consistency, cascade, all.",
)
@click.option(
    "--dataset",
    required=True,
    help="Path to test JSONL dataset.",
)
@click.option(
    "--embedder",
    required=True,
    type=click.Choice(["small", "medium", "large"]),
    help="Embedder tier to use.",
)
@click.option(
    "--detector-config",
    default=None,
    help="JSON file path or inline JSON string with detector hyperparameters.",
)
@click.option(
    "--threshold-config",
    default=None,
    help="Path to calibrated threshold JSON file (from bench.py calibrate).",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Cache embeddings across techniques.",
)
@click.option(
    "--threat-bank",
    default=None,
    help="Path to threat bank JSONL (default: data/processed/threat_bank.jsonl).",
)
@click.option(
    "--output",
    default=None,
    help="Output path: .json for single technique, .csv for all techniques.",
)
@click.option(
    "--mode",
    "benchmark_mode",
    default="full",
    show_default=True,
    type=click.Choice(["full", "fast"]),
    help="Benchmark execution mode. 'full' preserves current CLI profiling; 'fast' uses smaller profiling samples.",
)
def run(
    technique: str,
    dataset: str,
    embedder: str,
    detector_config: str | None,
    threshold_config: str | None,
    cache: bool,
    threat_bank: str | None,
    output: str | None,
    benchmark_mode: str,
) -> None:
    """Run detection technique(s) on DATASET and report metrics.

    When TECHNIQUE is "all", runs all four techniques and writes a combined
    CSV.  Otherwise runs a single technique and writes JSON results.

    \b
    Examples:
      bench.py run --technique threat_prototype --dataset data/processed/deepset_test.jsonl --embedder small
      bench.py run --technique all --dataset data/processed/spml_test.jsonl --embedder small --output results/all.csv
      bench.py run --technique cascade --dataset data/processed/synthetic_mismatch_test.jsonl --embedder small
    """
    import numpy as np

    from tadbench.data.schema import validate_jsonl, validate_sample
    from tadbench.embedders.cache import EmbeddingCache
    from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
    from tadbench.evaluation.calibration import load_thresholds
    from tadbench.evaluation.metrics import compute_all_metrics
    from tadbench.utils.io import read_jsonl

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise click.ClickException(f"Dataset not found: {dataset_path}")

    bank_path = (
        Path(threat_bank) if threat_bank is not None else PROCESSED_DIR / "threat_bank.jsonl"
    )

    # Resolve techniques
    if technique.strip().lower() == "all":
        techniques = ALL_TECHNIQUES
        run_all = True
    else:
        techniques = [technique.strip()]
        run_all = False

    # Resolve output path
    if output is not None:
        out_path = Path(output)
    elif run_all:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"run_all_{embedder}_{dataset_path.stem}.csv"
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"run_{techniques[0]}_{embedder}_{dataset_path.stem}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse detector config
    det_config = _load_detector_config(detector_config)

    # Load threshold config
    thresholds: dict[str, float] = {}
    calibrated_configs: dict[str, dict] = {}
    if threshold_config is not None:
        try:
            payload = load_thresholds(threshold_config)
            thresholds = payload.get("thresholds", {})
            calibrated_configs = payload.get("detector_configs", {})
            click.echo(f"[run] Loaded {len(thresholds)} thresholds from {threshold_config}")
        except Exception as exc:
            click.echo(f"[run] WARNING: could not load threshold config: {exc}")

    click.echo(f"[run] Dataset:    {dataset_path}")
    click.echo(f"[run] Embedder:   {embedder}")
    click.echo(f"[run] Techniques: {techniques}")
    click.echo(f"[run] Output:     {out_path}")
    click.echo(f"[run] Mode:       {benchmark_mode}")

    # Validate and load dataset
    click.echo("\n[run] Validating dataset ...")
    valid_count, errors = validate_jsonl(dataset_path)
    if errors:
        click.echo(f"[run] WARNING: {len(errors)} schema errors (first 5):")
        for e in errors[:5]:
            click.echo(f"  {e}")
    click.echo(f"[run] {valid_count:,} valid samples")

    raw = read_jsonl(dataset_path)
    if not raw:
        raise click.ClickException("Dataset is empty.")

    samples = []
    for row in raw:
        try:
            samples.append(validate_sample(row))
        except Exception:
            pass

    if not samples:
        raise click.ClickException("No valid samples after schema validation.")

    y_true = np.array([1 if s.label == "malicious" else 0 for s in samples])
    groups = [s.attack_type for s in samples]

    for tech in techniques:
        try:
            validate_dataset_for_technique(tech, samples, dataset_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    # Load embedder
    try:
        click.echo(f"\n[run] Loading embedder '{embedder}' ...")
        base_emb = SentenceTransformerEmbedder(embedder)
        emb = EmbeddingCache(base_emb) if cache else base_emb
    except Exception as exc:
        raise click.ClickException(f"Failed to load embedder: {exc}") from exc

    # Run each technique
    all_results: list[dict[str, Any]] = []
    profiling_cfg = RUN_PROFILING_MODES[benchmark_mode]

    for tech in techniques:
        click.echo(f"\n[run] === Technique: {tech} ===")
        merged_config = {**DEFAULT_DETECTOR_CONFIG.get(tech, {}),
                         **calibrated_configs.get(tech, {}), **det_config}

        try:
            from tadbench.cli.calibrate_cmd import _build_detector, _fit_detector

            detector = _build_detector(tech, emb, bank_path, merged_config)
            _fit_detector(detector, tech, bank_path, raw)

            click.echo(f"[run] Scoring {len(samples):,} samples ...")
            y_score = detector.score(samples)
            from tadbench.evaluation.efficiency import LatencyProfiler
            efficiency = LatencyProfiler().profile_detector(
                detector, samples,
                n_runs=int(profiling_cfg["latency_runs"] or 1),
                max_latency_samples=profiling_cfg["max_latency_samples"],
                max_throughput_samples=profiling_cfg["max_throughput_samples"],
            )
            p50, p95, p99 = efficiency.p50_ms, efficiency.p95_ms, efficiency.p99_ms
            throughput = efficiency.throughput_rps

            # Compute metrics
            threshold = thresholds.get(tech, .5 if tech == "cascade" else None)
            metrics = compute_all_metrics(y_true, y_score, groups=groups, threshold=threshold)

            result: dict[str, Any] = {
                "detector_name": tech,
                "embedder": embedder,
                "dataset": str(dataset_path),
                "n_samples": len(samples),
                "benchmark_mode": benchmark_mode,
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "tpr_at_fpr": metrics.get("tpr_at_fpr", {}),
                "calibration_error": metrics.get("calibration_error"),
                "per_group_metrics": metrics.get("per_group", {}),
                "per_attack_recall": metrics.get("per_attack_recall", {}),
                "operating_point": metrics.get("operating_point"),
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "throughput_rps": throughput,
                "rss_mb": efficiency.rss_mb,
                "rss_measurement": efficiency.rss_measurement,
                "embedding_calls": efficiency.embedding_calls_per_sample,
                "total_tokens": efficiency.total_tokens_embedded,
                "token_count_method": efficiency.token_count_method,
                "cross_encoder_calls": efficiency.cross_encoder_calls_per_sample,
                "latency_profile_samples": efficiency.latency_profile_samples,
                "throughput_profile_samples": efficiency.throughput_profile_samples,
            }

            # Cascade-specific fields
            if tech == "cascade" and hasattr(detector, "score_with_metadata"):
                try:
                    _, cascade_meta = detector.score_with_metadata(samples)
                    stages = [m.stage_reached for m in cascade_meta]
                    result["avg_stage_reached"] = float(np.mean(stages))
                    result["pct_filtered_stage1"] = float(np.mean([s == 1 for s in stages]))
                    result["pct_filtered_stage2"] = float(
                        np.mean([s <= 2 for s in stages])
                    )
                except Exception as e:
                    click.echo(f"[run]   cascade metadata unavailable: {e}")

            all_results.append(result)

            # Print per-technique summary
            click.echo(f"[run]   AUROC:      {result['auroc']:.4f}" if result["auroc"] else "[run]   AUROC: N/A")
            click.echo(f"[run]   AUPRC:      {result['auprc']:.4f}" if result["auprc"] else "[run]   AUPRC: N/A")
            click.echo(f"[run]   p50/p95/p99 latency (ms): {p50:.2f} / {p95:.2f} / {p99:.2f}")
            click.echo(f"[run]   Throughput: {throughput:.1f} samples/sec")

            if result.get("per_attack_recall"):
                click.echo("[run]   Per-attack recall:")
                for att, recall in sorted(result["per_attack_recall"].items()):
                    click.echo(f"    {att:<30} recall={recall:.4f}")

        except Exception as exc:
            click.echo(f"[run] ERROR running {tech}: {exc}")
            import traceback
            click.echo(traceback.format_exc())

    if not all_results:
        raise click.ClickException("No techniques produced results.")

    # Write output
    if run_all:
        _write_csv(all_results, out_path)
        click.echo(f"\n[run] Combined CSV: {out_path}  ({len(all_results)} rows)")
    else:
        result = all_results[0]
        out_path.write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"\n[run] Results JSON: {out_path}")

    # Final summary table
    click.echo("\n=== Summary ===")
    click.echo(f"{'Technique':<35} {'AUROC':>7} {'AUPRC':>7} {'p50ms':>7} {'RPS':>8}")
    click.echo("-" * 70)
    for r in all_results:
        auroc = f"{r['auroc']:.4f}" if r["auroc"] is not None else "  N/A "
        auprc = f"{r['auprc']:.4f}" if r["auprc"] is not None else "  N/A "
        click.echo(
            f"{r['detector_name']:<35} {auroc:>7} {auprc:>7} "
            f"{r['p50_ms']:>6.2f} {r['throughput_rps']:>8.1f}"
        )


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write a flat CSV from a list of result dicts."""
    import csv

    # Flatten tpr_at_fpr and per_attack_recall for CSV output
    rows = []
    for r in results:
        flat: dict[str, Any] = {
            "detector_name": r["detector_name"],
            "embedder": r["embedder"],
            "dataset": r["dataset"],
            "n_samples": r["n_samples"],
            "benchmark_mode": r.get("benchmark_mode"),
            "auroc": r.get("auroc"),
            "auprc": r.get("auprc"),
            "calibration_error": r.get("calibration_error"),
            "p50_ms": r.get("p50_ms"),
            "p95_ms": r.get("p95_ms"),
            "p99_ms": r.get("p99_ms"),
            "throughput_rps": r.get("throughput_rps"),
            "rss_mb": r.get("rss_mb"),
            "embedding_calls": r.get("embedding_calls"),
            "total_tokens": r.get("total_tokens"),
            "cross_encoder_calls": r.get("cross_encoder_calls", 0),
            "latency_profile_samples": r.get("latency_profile_samples"),
            "throughput_profile_samples": r.get("throughput_profile_samples"),
        }
        # TPR@FPR columns
        for fpr_key, tpr_val in (r.get("tpr_at_fpr") or {}).items():
            flat[f"tpr_at_fpr_{fpr_key}"] = tpr_val
        # Per-attack AUROC
        for att, rec_dict in (r.get("per_group_metrics") or {}).items():
            if isinstance(rec_dict, dict):
                flat[f"auroc_{att}"] = rec_dict.get("auroc")
        for att, recall in (r.get("per_attack_recall") or {}).items():
            flat[f"recall_{att}"] = recall
        for key, value in (r.get("operating_point") or {}).items():
            flat[f"operating_{key}"] = value
        flat["token_count_method"] = r.get("token_count_method")
        flat["rss_measurement"] = r.get("rss_measurement")
        # Cascade-specific
        for key in ("avg_stage_reached", "pct_filtered_stage1", "pct_filtered_stage2"):
            if key in r:
                flat[key] = r[key]
        rows.append(flat)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    # Collect any extra keys from other rows
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
