"""CLI command: bench.py calibrate — calibrate detection thresholds on val data."""

from __future__ import annotations

from pathlib import Path

import click

from tadbench.compat import validate_dataset_for_technique
from tadbench.config import DEFAULT_DETECTOR_CONFIG, PROCESSED_DIR, THRESHOLDS_DIR


# All four techniques that can be calibrated
ALL_TECHNIQUES = ["threat_prototype", "intra_prompt_outlier", "intent_content_consistency", "cascade"]


@click.command("calibrate")
@click.option(
    "--dataset",
    required=True,
    help="Path to validation JSONL dataset.",
)
@click.option(
    "--embedder",
    required=True,
    type=click.Choice(["small", "medium", "large"]),
    help="Embedder tier to use.",
)
@click.option(
    "--technique",
    default="all",
    show_default=True,
    help='Technique name or "all". One of: threat_prototype, intra_prompt_outlier, '
         "intent_content_consistency, cascade.",
)
@click.option(
    "--target-fpr",
    default=0.005,
    show_default=True,
    type=float,
    help="Target false positive rate for threshold calibration.",
)
@click.option(
    "--threat-bank",
    default=None,
    help="Path to threat bank JSONL (required for threat_prototype / cascade).",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save threshold JSON files (default: thresholds/).",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Cache embeddings across techniques.",
)
def calibrate(
    dataset: str,
    embedder: str,
    technique: str,
    target_fpr: float,
    threat_bank: str | None,
    output_dir: str | None,
    cache: bool,
) -> None:
    """Calibrate detection thresholds on a validation dataset.

    Scores each technique on the validation data and finds the threshold
    that achieves TARGET_FPR.  Results are saved to OUTPUT_DIR as JSON.
    """
    import numpy as np

    from tadbench.data.schema import validate_jsonl
    from tadbench.embedders.cache import EmbeddingCache
    from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
    from tadbench.evaluation.calibration import calibrate_threshold, save_thresholds
    from tadbench.utils.io import read_jsonl

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise click.ClickException(f"Dataset not found: {dataset_path}")

    out_dir = Path(output_dir) if output_dir is not None else THRESHOLDS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    bank_path = (
        Path(threat_bank) if threat_bank is not None else PROCESSED_DIR / "threat_bank.jsonl"
    )

    # Resolve techniques list
    if technique.strip().lower() == "all":
        techniques = ALL_TECHNIQUES
    else:
        techniques = [technique.strip()]

    click.echo(f"[calibrate] Dataset:    {dataset_path}")
    click.echo(f"[calibrate] Embedder:   {embedder}")
    click.echo(f"[calibrate] Techniques: {techniques}")
    click.echo(f"[calibrate] Target FPR: {target_fpr}")
    click.echo(f"[calibrate] Output dir: {out_dir}")

    # Validate dataset
    click.echo("\n[calibrate] Validating dataset ...")
    valid_count, errors = validate_jsonl(dataset_path)
    if errors:
        click.echo(f"[calibrate] WARNING: {len(errors)} validation errors in dataset:")
        for err in errors[:5]:
            click.echo(f"  {err}")
    click.echo(f"[calibrate] {valid_count:,} valid samples")

    # Load samples
    raw_samples = read_jsonl(dataset_path)
    if not raw_samples:
        raise click.ClickException("Dataset is empty.")

    from tadbench.data.schema import validate_sample

    samples = []
    for row in raw_samples:
        try:
            samples.append(validate_sample(row))
        except Exception:
            pass

    if not samples:
        raise click.ClickException("No valid samples after schema validation.")

    for tech in techniques:
        try:
            validate_dataset_for_technique(tech, samples, dataset_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    # Build y_true
    y_true = np.array([1 if s.label == "malicious" else 0 for s in samples])

    # Initialise embedder (with optional cache)
    try:
        click.echo(f"\n[calibrate] Loading embedder '{embedder}' ...")
        base_emb = SentenceTransformerEmbedder(embedder)
        emb = EmbeddingCache(base_emb) if cache else base_emb
    except Exception as exc:
        raise click.ClickException(f"Failed to load embedder: {exc}") from exc

    # Score each technique
    calibrated: dict[str, float] = {}

    for tech in techniques:
        click.echo(f"\n[calibrate] Scoring technique: {tech} ...")
        try:
            detector = _build_detector(tech, emb, bank_path, DEFAULT_DETECTOR_CONFIG.get(tech, {}))
            _fit_detector(detector, tech, bank_path, raw_samples)
            y_score = detector.score(samples)

            threshold = calibrate_threshold(y_true[: len(y_score)], y_score, target_fpr)
            calibrated[tech] = threshold
            click.echo(f"[calibrate]   {tech}: threshold = {threshold:.6f}")

        except Exception as exc:
            click.echo(f"[calibrate]   {tech}: FAILED — {exc}")

    if not calibrated:
        raise click.ClickException("No techniques were successfully calibrated.")

    # Save per-embedder threshold file
    out_path = out_dir / f"{embedder}_fpr_{target_fpr:.4f}.json".replace("0.", "0")
    # Normalise filename: e.g. 0.005 -> small_fpr_0.0050.json
    out_path = out_dir / f"{embedder}_fpr_{target_fpr}.json"
    save_thresholds(
        calibrated,
        out_path,
        target_fpr=target_fpr,
        dataset_name=dataset_path.name,
    )

    click.echo(f"\n[calibrate] Saved {len(calibrated)} threshold(s) to: {out_path}")
    click.echo("\nSummary:")
    click.echo(f"  {'Technique':<35} {'Threshold':>12}")
    click.echo(f"  {'-'*35} {'-'*12}")
    for tech, thresh in calibrated.items():
        click.echo(f"  {tech:<35} {thresh:>12.6f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_detector(technique: str, emb, bank_path: Path, config: dict):
    """Instantiate a detector from a technique name and config dict."""
    if technique == "threat_prototype":
        from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
        return ThreatPrototypeDetector(
            emb,
            index_type=config.get("index_type", "flat"),
            top_k=config.get("top_k", 1),
            aggregation=config.get("aggregation", "max"),
        )
    elif technique == "intra_prompt_outlier":
        from tadbench.detectors.intra_prompt_outlier import IntraPromptOutlierDetector
        return IntraPromptOutlierDetector(
            emb,
            outlier_detector=config.get("outlier_detector", "lof"),
            chunker=config.get("chunker", "sentence"),
            aggregation=config.get("aggregation", "max"),
            min_chunks=config.get("min_chunks", 5),
        )
    elif technique == "intent_content_consistency":
        from tadbench.detectors.intent_content import IntentContentDetector
        return IntentContentDetector(
            emb,
            scorer=config.get("scorer", "bi_encoder"),
            body_strategy=config.get("body_strategy", "full"),
        )
    elif technique == "cascade":
        from tadbench.detectors.cascade import CascadeDetector
        return CascadeDetector(
            emb,
            stage1_threshold=config.get("stage1_threshold", 0.8),
            stage2_threshold=config.get("stage2_threshold", 0.7),
            stage3_threshold=config.get("stage3_threshold", 0.5),
            tps_kwargs=config.get("tps_kwargs"),
            ipsod_kwargs=config.get("ipsod_kwargs"),
            iccd_kwargs=config.get("iccd_kwargs"),
        )
    else:
        raise ValueError(f"Unknown technique: {technique!r}")


def _fit_detector(detector, technique: str, bank_path: Path, raw_samples: list) -> None:
    """Fit a detector with technique-specific benchmark semantics."""
    from tadbench.data.schema import validate_sample

    samples = []
    for row in raw_samples:
        try:
            samples.append(validate_sample(row))
        except Exception:
            pass

    if technique in ("threat_prototype", "cascade"):
        from tadbench.data.validation import load_threat_bank
        bank = load_threat_bank(bank_path, samples)
        if technique == "threat_prototype":
            detector.fit(bank)
        else:
            detector._stage1.fit(bank)
            detector._stage2.fit(samples)
            detector._stage3.fit(samples)
    else:
        detector.fit(samples)
