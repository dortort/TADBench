"""CLI command: bench.py matrix — run the full benchmark matrix from a config file."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tadbench.config import RESULTS_DIR


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
def matrix(
    config: str,
    output: str | None,
    embedder: str | None,
    dry_run: bool,
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

    # Apply embedder override
    if embedder is not None:
        matrix_cfg["embedders"] = [embedder]
        click.echo(f"[matrix] Embedder override: {embedder}")

    embedders = matrix_cfg.get("embedders", ["small"])
    techniques = matrix_cfg.get("techniques", [])
    datasets = matrix_cfg.get("datasets", [])
    target_fprs = matrix_cfg.get("target_fprs", [0.005])

    click.echo(f"[matrix] Config:     {config_path}")
    click.echo(f"[matrix] Embedders:  {embedders}")
    click.echo(f"[matrix] Datasets:   {datasets}")
    click.echo(f"[matrix] Techniques: {[t['name'] for t in techniques]}")
    click.echo(f"[matrix] Output:     {out_path}")

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

    # Run the matrix
    try:
        from tadbench.runner import BenchmarkRunner
        runner = BenchmarkRunner(
            embedder_tier=embedders[0],  # runner is re-created per embedder below
            cache=True,
            results_dir=str(out_path.parent),
        )
        df = runner.run_matrix(str(config_path))
    except (ImportError, AttributeError):
        # runner.py not yet implemented — use CLI-level implementation
        df = _run_matrix_inline(matrix_cfg, embedders, techniques, datasets, target_fprs)

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        raise click.ClickException("Matrix run produced no results.")

    # Save CSV
    try:
        df.to_csv(out_path, index=False)
        click.echo(f"\n[matrix] Results written to: {out_path}")
        click.echo(f"[matrix] {len(df):,} rows x {len(df.columns)} columns")
    except AttributeError:
        # df might be a list of dicts if pandas unavailable
        import csv
        if df:
            with out_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(df[0].keys()))
                writer.writeheader()
                writer.writerows(df)
        click.echo(f"\n[matrix] Results written to: {out_path}")
        click.echo(f"[matrix] {len(df)} rows")


def _run_matrix_inline(
    matrix_cfg: dict,
    embedders: list[str],
    techniques: list[dict],
    datasets: list[str],
    target_fprs: list[float],
):
    """Fallback inline matrix runner when BenchmarkRunner is unavailable."""
    import subprocess
    import sys

    all_rows: list[dict] = []

    for emb_tier in embedders:
        for tech in techniques:
            tech_name = tech["name"]
            for ds in datasets:
                ds_path = Path(ds)
                if not ds_path.exists():
                    click.echo(f"[matrix]   SKIP (not found): {ds}")
                    continue

                click.echo(f"\n[matrix] Running: {emb_tier} / {tech_name} / {ds_path.name}")
                try:
                    result = subprocess.run(
                        [
                            sys.executable, "bench.py", "run",
                            "--technique", tech_name,
                            "--dataset", str(ds_path),
                            "--embedder", emb_tier,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode == 0:
                        click.echo(f"[matrix]   OK")
                    else:
                        click.echo(f"[matrix]   FAILED (exit {result.returncode})")
                        click.echo(result.stderr[-500:] if result.stderr else "")
                except subprocess.TimeoutExpired:
                    click.echo(f"[matrix]   TIMEOUT")
                except Exception as exc:
                    click.echo(f"[matrix]   ERROR: {exc}")

    return all_rows
