"""CLI command: bench.py prepare — transform, generate, and split datasets."""

from __future__ import annotations

from pathlib import Path

import click

from tadbench.config import PROCESSED_DIR, RAW_DIR


@click.command("prepare")
@click.option(
    "--output-dir",
    default=None,
    help="Output directory for processed data (default: data/processed).",
)
@click.option(
    "--raw-dir",
    default=None,
    help="Directory containing raw downloaded data (default: data/raw).",
)
@click.option(
    "--threat-bank-sources",
    default="all",
    show_default=True,
    help='Comma-separated list of sources for threat bank, or "all".',
)
@click.option(
    "--include-adversarial/--no-adversarial",
    default=True,
    show_default=True,
    help="Generate adversarial evasion samples.",
)
def prepare(
    output_dir: str | None,
    raw_dir: str | None,
    threat_bank_sources: str,
    include_adversarial: bool,
) -> None:
    """Transform raw datasets, generate synthetic samples, and build splits.

    Runs the full data preparation pipeline:

    \b
    1. Transform raw HuggingFace datasets -> unified JSONL
    2. Generate synthetic mismatch dataset (Technique 3)
    3. Optionally generate adversarial evasion samples
    4. Build train/val/test splits, threat bank, and combined sets
    """
    from tadbench.data.generate_adversarial import generate_adversarial
    from tadbench.data.generate_synthetic import generate_synthetic
    from tadbench.data.splits import build_splits
    from tadbench.data.transform import transform_all

    raw = Path(raw_dir) if raw_dir is not None else RAW_DIR
    out = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Resolve threat-bank sources
    if threat_bank_sources.strip().lower() == "all":
        sources = None  # build_splits / build_threat_bank defaults to all three
    else:
        sources = [s.strip() for s in threat_bank_sources.split(",") if s.strip()]

    try:
        # Step 1: transform raw datasets
        click.echo("\n=== Step 1/4: Transforming raw datasets ===")
        transform_all(raw_dir=raw, output_dir=out)

        # Step 2: generate synthetic mismatch dataset
        click.echo("\n=== Step 2/4: Generating synthetic mismatch dataset ===")
        generate_synthetic(output_path=out / "synthetic_mismatch.jsonl")

        # Step 3: generate adversarial evasion samples
        if include_adversarial:
            click.echo("\n=== Step 3/4: Generating adversarial evasion samples ===")
            generate_adversarial(output_dir=out)
        else:
            click.echo("\n=== Step 3/4: Skipping adversarial generation (--no-adversarial) ===")

        # Step 4: build splits, threat bank, combined sets
        click.echo("\n=== Step 4/4: Building splits and combined sets ===")
        build_splits(processed_dir=out, threat_bank_sources=sources)

    except Exception as exc:
        raise click.ClickException(f"Prepare failed: {exc}") from exc

    click.echo(f"\n[prepare] Done. Processed data in: {out}")

    # Summary stats
    jsonl_files = sorted(out.glob("*.jsonl"))
    if jsonl_files:
        click.echo(f"[prepare] {len(jsonl_files)} JSONL files created:")
        for f in jsonl_files:
            size_kb = f.stat().st_size // 1024
            click.echo(f"  {f.name}  ({size_kb} KB)")
