"""CLI command: bench.py drift-eval — evaluate recall degradation on held-out attack types."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tadbench.config import PROCESSED_DIR, RESULTS_DIR


@click.command("drift-eval")
@click.option(
    "--holdout-attack",
    required=True,
    help="Attack type to hold out from training (e.g. jailbreak, role_override).",
)
@click.option(
    "--embedder",
    required=True,
    type=click.Choice(["small", "medium", "large"]),
    help="Embedder tier to use.",
)
@click.option(
    "--technique",
    default="threat_prototype",
    show_default=True,
    help="Detection technique to evaluate.",
)
@click.option(
    "--dataset",
    default=None,
    help="Path to JSONL dataset (default: data/processed/benchmark_combined.jsonl).",
)
@click.option(
    "--target-fpr",
    default=0.005,
    show_default=True,
    type=float,
    help="Target FPR for threshold calibration.",
)
@click.option(
    "--output",
    default=None,
    help="Output JSON path (default: results/drift_{holdout}_{embedder}.json).",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Cache embeddings.",
)
def drift_eval(
    holdout_attack: str,
    embedder: str,
    technique: str,
    dataset: str | None,
    target_fpr: float,
    output: str | None,
    cache: bool,
) -> None:
    """Evaluate recall degradation when a novel attack type is withheld from training.

    Fits and calibrates the detector WITHOUT the held-out attack type, then
    evaluates on a test set that includes it.  Produces a drift report JSON
    with recall_degradation and threshold_drift.

    \b
    Example:
      bench.py drift-eval --holdout-attack jailbreak --embedder small
    """
    from tadbench.embedders.cache import EmbeddingCache
    from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
    from tadbench.evaluation.drift import DriftEvaluator
    from tadbench.utils.io import read_jsonl

    ds_path = (
        Path(dataset) if dataset is not None
        else PROCESSED_DIR / "benchmark_combined.jsonl"
    )
    if not ds_path.exists():
        raise click.ClickException(
            f"Dataset not found: {ds_path}\n"
            "Run 'bench.py prepare' first, or specify --dataset."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(output) if output is not None
        else RESULTS_DIR / f"drift_{holdout_attack}_{embedder}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"[drift-eval] Dataset:        {ds_path}")
    click.echo(f"[drift-eval] Embedder:       {embedder}")
    click.echo(f"[drift-eval] Technique:      {technique}")
    click.echo(f"[drift-eval] Holdout attack: {holdout_attack}")
    click.echo(f"[drift-eval] Target FPR:     {target_fpr}")
    click.echo(f"[drift-eval] Output:         {out_path}")

    # Load all samples
    click.echo("\n[drift-eval] Loading dataset ...")
    raw = read_jsonl(ds_path)
    if not raw:
        raise click.ClickException("Dataset is empty.")

    # Check holdout attack exists
    attack_types_present = {r["attack_type"] for r in raw}
    if holdout_attack not in attack_types_present:
        raise click.ClickException(
            f"Holdout attack type '{holdout_attack}' not found in dataset.\n"
            f"Available attack types: {sorted(attack_types_present)}"
        )

    holdout_count = sum(1 for r in raw if r["attack_type"] == holdout_attack)
    if holdout_count < 10:
        click.echo(
            f"[drift-eval] WARNING: only {holdout_count} holdout samples "
            "(< 10 recommended for reliable evaluation)"
        )

    click.echo(f"[drift-eval] Total samples: {len(raw):,}")
    click.echo(f"[drift-eval] Holdout samples ({holdout_attack}): {holdout_count:,}")

    # Load embedder
    try:
        click.echo(f"\n[drift-eval] Loading embedder '{embedder}' ...")
        base_emb = SentenceTransformerEmbedder(embedder)
        emb = EmbeddingCache(base_emb) if cache else base_emb
    except Exception as exc:
        raise click.ClickException(f"Failed to load embedder: {exc}") from exc

    # Build detector
    try:
        from tadbench.cli.calibrate_cmd import _build_detector
        from tadbench.config import DEFAULT_DETECTOR_CONFIG

        detector = _build_detector(
            technique, emb,
            PROCESSED_DIR / "threat_bank.jsonl",
            DEFAULT_DETECTOR_CONFIG.get(technique, {}),
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to build detector: {exc}") from exc

    # Run drift evaluation
    try:
        click.echo("\n[drift-eval] Running drift evaluation ...")
        evaluator = DriftEvaluator(holdout_attack_type=holdout_attack)
        train, val, test = evaluator.create_drift_splits(raw)

        click.echo(
            f"[drift-eval] Splits — train: {len(train):,}, "
            f"val: {len(val):,}, test: {len(test):,}"
        )

        report = evaluator.evaluate_drift(
            detector=detector,
            train=train,
            val=val,
            test=test,
            target_fpr=target_fpr,
        )

    except Exception as exc:
        raise click.ClickException(f"Drift evaluation failed: {exc}") from exc

    # Serialise report to JSON
    try:
        report_dict = {
            "holdout_attack_type": report.holdout_attack_type,
            "recall_on_known_attacks": report.recall_on_known_attacks,
            "recall_on_holdout_attack": report.recall_on_holdout_attack,
            "recall_degradation": report.recall_degradation,
            "threshold_before": report.threshold_before,
            "threshold_after_retrain": report.threshold_after_retrain,
            "threshold_drift": report.threshold_drift,
            "retraining_cost_seconds": report.retraining_cost_seconds,
            "embedder": embedder,
            "technique": technique,
            "dataset": str(ds_path),
            "target_fpr": target_fpr,
        }
    except AttributeError:
        # report might already be a dict
        report_dict = report if isinstance(report, dict) else {}

    out_path.write_text(json.dumps(report_dict, indent=2, default=str))

    click.echo(f"\n[drift-eval] Report written to: {out_path}")
    click.echo("\n=== Drift Report ===")
    for key, val in report_dict.items():
        if isinstance(val, float):
            click.echo(f"  {key:<35} {val:.6f}")
        else:
            click.echo(f"  {key:<35} {val}")
