"""CLI command: bench.py matrix — run the full benchmark matrix from a config file."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tadbench.config import PROCESSED_DIR, RESULTS_DIR


@click.command("matrix")
@click.option(
    "--config",
    default="matrix_config.json",
    show_default=True,
    help="Path to matrix configuration JSON file.",
)
@click.option(
    "--output",
    default=None,
    help="Output CSV path (default: results/matrix_results.csv).",
)
@click.option(
    "--embedder",
    default=None,
    type=click.Choice(["small", "medium", "large"]),
    help="Override embedder tier (overrides config file).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be run without executing.",
)
@click.option(
    "--mode",
    "benchmark_mode",
    default="full",
    show_default=True,
    type=click.Choice(["full", "fast"]),
    help="Benchmark execution mode. 'full' preserves exhaustive profiling; 'fast' uses sampled efficiency estimates.",
)
@click.option(
    "--threat-bank",
    default=None,
    help="Path to threat bank JSONL (default: data/processed/threat_bank.jsonl).",
)
def matrix(
    config: str,
    output: str | None,
    embedder: str | None,
    dry_run: bool,
    benchmark_mode: str,
    threat_bank: str | None,
) -> None:
    """Run the full benchmark matrix as defined in CONFIG.

    The config file specifies embedders, techniques, variants, datasets,
    and evaluation options.  Each (technique x embedder x dataset) combination
    is run and results are aggregated into a single CSV.

    \b
    Example:
      bench.py matrix --config matrix_config.json --output results/matrix.csv
    """
    config_path = Path(config)
    if not config_path.exists():
        raise click.ClickException(f"Matrix config not found: {config_path}")

    try:
        matrix_cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in config file: {exc}") from exc

    out_path = Path(output) if output is not None else RESULTS_DIR / "matrix_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank_path = (
        Path(threat_bank) if threat_bank is not None else PROCESSED_DIR / "threat_bank.jsonl"
    )

    # Apply embedder override
    if embedder is not None:
        matrix_cfg["embedders"] = [embedder]
        click.echo(f"[matrix] Embedder override: {embedder}")

    embedders = matrix_cfg.get("embedders", ["small"])
    techniques = matrix_cfg.get("techniques", [])
    datasets = matrix_cfg.get("datasets", [])
    click.echo(f"[matrix] Config:     {config_path}")
    click.echo(f"[matrix] Embedders:  {embedders}")
    click.echo(f"[matrix] Datasets:   {datasets}")
    click.echo(f"[matrix] Techniques: {[t['name'] for t in techniques]}")
    click.echo(f"[matrix] Output:     {out_path}")
    click.echo(f"[matrix] Mode:       {benchmark_mode}")
    click.echo(f"[matrix] Threat bank: {bank_path}")

    total_runs = sum(
        len(embedders) * len(t.get("variants", [{}])) * len(datasets)
        for t in techniques
    )
    click.echo(f"[matrix] Total runs: {total_runs}")

    if dry_run:
        click.echo("\n[matrix] DRY RUN — no execution.")
        for emb_tier in embedders:
            for tech in techniques:
                for variant in tech.get("variants", [{}]):
                    for ds in datasets:
                        click.echo(
                            f"  {emb_tier} / {tech['name']} / "
                            f"{json.dumps(variant)} / {ds}"
                        )
        return

    if not bank_path.exists():
        raise click.ClickException(
            f"Threat bank not found: {bank_path}. Run 'tadbench prepare' or pass --threat-bank."
        )

    from tadbench.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        embedder_tier=embedders[0],
        cache=True,
        results_dir=str(out_path.parent),
    )
    df = runner.run_matrix(
        str(config_path),
        benchmark_mode=benchmark_mode,
        threat_bank_path=str(bank_path),
        embedders=embedders,
    )

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        raise click.ClickException("Matrix run produced no results.")

    df.to_csv(out_path, index=False)
    click.echo(f"\n[matrix] Results written to: {out_path}")
    click.echo(f"[matrix] {len(df):,} rows x {len(df.columns)} columns")
