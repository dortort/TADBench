"""CLI command: bench.py build-index — build a FAISS threat-prototype index."""

from __future__ import annotations

from pathlib import Path

import click

from tadbench.config import INDEX_DIR, PROCESSED_DIR


@click.command("build-index")
@click.option(
    "--embedder",
    required=True,
    type=click.Choice(["small", "medium", "large"]),
    help="Embedder tier to use.",
)
@click.option(
    "--index-type",
    default="flat",
    show_default=True,
    type=click.Choice(["flat", "hnsw", "ivf"]),
    help="FAISS index type.",
)
@click.option(
    "--threat-bank",
    default=None,
    help="Path to threat bank JSONL (default: data/processed/threat_bank.jsonl).",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save the index (default: indexes/).",
)
def build_index(
    embedder: str,
    index_type: str,
    threat_bank: str | None,
    output_dir: str | None,
) -> None:
    """Build a FAISS index from the threat bank for EMBEDDER tier."""
    from tadbench.detectors.threat_prototype import ThreatPrototypeDetector
    from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder

    bank_path = Path(threat_bank) if threat_bank is not None else PROCESSED_DIR / "threat_bank.jsonl"
    idx_dir = Path(output_dir) if output_dir is not None else INDEX_DIR
    idx_dir.mkdir(parents=True, exist_ok=True)

    index_filename = f"{embedder}_{index_type}.index"
    index_path = idx_dir / index_filename

    click.echo(f"[build-index] Embedder:     {embedder}")
    click.echo(f"[build-index] Index type:   {index_type}")
    click.echo(f"[build-index] Threat bank:  {bank_path}")
    click.echo(f"[build-index] Output:       {index_path}")

    if not bank_path.exists():
        raise click.ClickException(
            f"Threat bank not found: {bank_path}\n"
            "Run 'bench.py prepare' first to generate it."
        )

    try:
        click.echo(f"\n[build-index] Loading embedder '{embedder}' ...")
        emb = SentenceTransformerEmbedder(embedder)

        click.echo(f"[build-index] Building {index_type} index ...")
        detector = ThreatPrototypeDetector(emb, index_type=index_type)
        detector.build_index(bank_path)

        click.echo(f"[build-index] Saving index to {index_path} ...")
        detector.save_index(index_path)

    except Exception as exc:
        raise click.ClickException(f"build-index failed: {exc}") from exc

    meta_path = index_path.with_suffix("").with_suffix(".meta.json")
    click.echo("\n[build-index] Done.")
    click.echo(f"  Index:    {index_path}  ({index_path.stat().st_size // 1024} KB)")
    if meta_path.exists():
        click.echo(f"  Metadata: {meta_path}")
        import json
        meta = json.loads(meta_path.read_text())
        click.echo(f"  Vectors:  {meta.get('sample_count', '?'):,}")
        click.echo(f"  Embedder: {meta.get('embedder', '?')}")
        click.echo(f"  Built at: {meta.get('build_timestamp', '?')}")
