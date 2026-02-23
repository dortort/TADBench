"""CLI command: bench.py download — download all public datasets."""

from __future__ import annotations

from pathlib import Path

import click

from tadbench.config import RAW_DIR


@click.command("download")
@click.option(
    "--output-dir",
    default=None,
    show_default=True,
    help="Output directory for raw data (default: data/raw).",
)
def download(output_dir: str | None) -> None:
    """Download all benchmark datasets from HuggingFace to OUTPUT_DIR."""
    from tadbench.data.download import download_all

    out = Path(output_dir) if output_dir is not None else RAW_DIR
    click.echo(f"[download] Saving raw datasets to: {out}")

    try:
        download_all(out)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Download failed: {exc}") from exc

    click.echo(f"\n[download] Done. Raw data in: {out}")
