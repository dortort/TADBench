"""CLI command: bench.py scaling — measure latency scaling curves."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tadbench.config import PROCESSED_DIR, RESULTS_DIR


ALL_TECHNIQUES = ["threat_prototype", "intra_prompt_outlier", "intent_content_consistency", "cascade"]


@click.command("scaling")
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
    help='Technique name or "all".',
)
@click.option(
    "--dataset",
    default=None,
    help="Path to JSONL dataset (default: data/processed/benchmark_combined.jsonl).",
)
@click.option(
    "--threat-bank",
    default=None,
    help="Path to threat bank JSONL (default: data/processed/threat_bank.jsonl).",
)
@click.option(
    "--output",
    default=None,
    help="Output JSON path (default: results/scaling_{embedder}.json).",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Cache embeddings.",
)
def scaling(
    embedder: str,
    technique: str,
    dataset: str | None,
    threat_bank: str | None,
    output: str | None,
    cache: bool,
) -> None:
    """Measure latency scaling curves for detection techniques.

    Produces per-technique scaling data:

    \b
    - Latency vs prompt length (token bins)
    - Latency vs threat bank size (threat_prototype only)
    - Latency vs chunks per prompt (intra_prompt_outlier only)
    - Latency vs pair length (intent_content_consistency only)

    \b
    Example:
      bench.py scaling --embedder small
      bench.py scaling --embedder small --technique threat_prototype
    """
    from tadbench.embedders.cache import EmbeddingCache
    from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
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

    bank_path = (
        Path(threat_bank) if threat_bank is not None
        else PROCESSED_DIR / "threat_bank.jsonl"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(output) if output is not None
        else RESULTS_DIR / f"scaling_{embedder}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve techniques
    if technique.strip().lower() == "all":
        techniques = ALL_TECHNIQUES
    else:
        techniques = [technique.strip()]

    click.echo(f"[scaling] Dataset:    {ds_path}")
    click.echo(f"[scaling] Embedder:   {embedder}")
    click.echo(f"[scaling] Techniques: {techniques}")
    click.echo(f"[scaling] Output:     {out_path}")

    # Load samples
    click.echo("\n[scaling] Loading dataset ...")
    raw = read_jsonl(ds_path)
    if not raw:
        raise click.ClickException("Dataset is empty.")
    click.echo(f"[scaling] {len(raw):,} samples loaded")

    # Validate and convert to BenchmarkSample
    from tadbench.data.schema import validate_sample
    samples = []
    for row in raw:
        try:
            samples.append(validate_sample(row))
        except Exception:
            pass
    click.echo(f"[scaling] {len(samples):,} valid samples after schema check")

    if not samples:
        raise click.ClickException("No valid samples after schema validation.")

    # Load embedder
    try:
        click.echo(f"\n[scaling] Loading embedder '{embedder}' ...")
        base_emb = SentenceTransformerEmbedder(embedder)
        emb = EmbeddingCache(base_emb) if cache else base_emb
    except Exception as exc:
        raise click.ClickException(f"Failed to load embedder: {exc}") from exc

    # Run scaling for each technique
    all_scaling: dict[str, dict] = {}

    for tech in techniques:
        click.echo(f"\n[scaling] === Technique: {tech} ===")
        try:
            result = _run_scaling_for_technique(
                tech, emb, samples, bank_path
            )
            all_scaling[tech] = result
            _print_scaling_summary(tech, result)
        except Exception as exc:
            click.echo(f"[scaling] ERROR for {tech}: {exc}")
            import traceback
            click.echo(traceback.format_exc())

    if not all_scaling:
        raise click.ClickException("No scaling data produced.")

    out_path.write_text(json.dumps(all_scaling, indent=2, default=str))
    click.echo(f"\n[scaling] Scaling data written to: {out_path}")


def _run_scaling_for_technique(
    technique: str,
    emb,
    samples: list,
    bank_path: Path,
) -> dict:
    """Run scaling measurements for a single technique."""
    import time

    import numpy as np

    from tadbench.cli.calibrate_cmd import _build_detector, _fit_detector
    from tadbench.config import DEFAULT_DETECTOR_CONFIG

    config = DEFAULT_DETECTOR_CONFIG.get(technique, {})
    detector = _build_detector(technique, emb, bank_path, config)

    raw_dicts = [
        {
            "id": s.id,
            "prompt_body": s.prompt_body,
            "advertised_description": s.advertised_description,
            "label": s.label,
            "attack_type": s.attack_type,
            "source": s.source,
            "metadata": s.metadata,
        }
        for s in samples
    ]
    _fit_detector(detector, technique, bank_path, raw_dicts)

    result: dict = {}

    # ---------------------------------------------------------------
    # 1. Latency vs prompt length (all techniques)
    # ---------------------------------------------------------------
    click.echo(f"[scaling]   Measuring latency vs prompt length ...")
    length_bins = {
        "short_0_128": [s for s in samples if _tokens(s) <= 128],
        "medium_129_512": [s for s in samples if 129 <= _tokens(s) <= 512],
        "long_513_2048": [s for s in samples if 513 <= _tokens(s) <= 2048],
        "very_long_2048+": [s for s in samples if _tokens(s) > 2048],
    }

    length_curve = []
    for bin_name, bin_samples in length_bins.items():
        if len(bin_samples) < 3:
            continue
        probe = bin_samples[:min(50, len(bin_samples))]
        latencies = _measure_latencies_ms(detector, probe)
        if latencies:
            length_curve.append({
                "bin": bin_name,
                "n_samples": len(probe),
                "avg_tokens": float(np.mean([_tokens(s) for s in probe])),
                "p50_ms": _pct(latencies, 50),
                "p95_ms": _pct(latencies, 95),
                "p99_ms": _pct(latencies, 99),
            })
            click.echo(
                f"[scaling]     {bin_name}: p50={_pct(latencies,50):.2f}ms "
                f"p95={_pct(latencies,95):.2f}ms  (n={len(probe)})"
            )

    result["latency_vs_prompt_length"] = length_curve

    # ---------------------------------------------------------------
    # 2. Latency vs threat bank size (threat_prototype / cascade only)
    # ---------------------------------------------------------------
    if technique in ("threat_prototype", "cascade") and bank_path.exists():
        click.echo(f"[scaling]   Measuring latency vs threat bank size ...")
        from tadbench.utils.io import read_jsonl

        bank_raw = read_jsonl(bank_path)
        bank_sizes = [100, 500, 1000, 5000, 10000, len(bank_raw)]
        bank_sizes = sorted(set(sz for sz in bank_sizes if sz <= len(bank_raw)))

        bank_curve = []
        probe_samples = samples[:min(30, len(samples))]

        for sz in bank_sizes:
            try:
                sub_bank = bank_raw[:sz]
                sub_detector = _build_detector(technique, emb, bank_path, config)

                from tadbench.data.schema import validate_sample
                bank_samples_obj = []
                for row in sub_bank:
                    try:
                        bank_samples_obj.append(validate_sample(row))
                    except Exception:
                        pass

                if bank_samples_obj:
                    sub_detector.fit(bank_samples_obj)
                    latencies = _measure_latencies_ms(sub_detector, probe_samples)
                    if latencies:
                        bank_curve.append({
                            "bank_size": sz,
                            "p50_ms": _pct(latencies, 50),
                            "p95_ms": _pct(latencies, 95),
                            "p99_ms": _pct(latencies, 99),
                        })
                        click.echo(
                            f"[scaling]     bank_size={sz}: "
                            f"p50={_pct(latencies,50):.2f}ms"
                        )
            except Exception as exc:
                click.echo(f"[scaling]     bank_size={sz}: ERROR — {exc}")

        result["latency_vs_threat_bank_size"] = bank_curve

    # ---------------------------------------------------------------
    # 3. Latency vs chunks per prompt (intra_prompt_outlier only)
    # ---------------------------------------------------------------
    if technique == "intra_prompt_outlier":
        click.echo(f"[scaling]   Measuring latency vs chunks per prompt ...")
        from tadbench.utils.chunker import SentenceChunker

        chunker = SentenceChunker()

        def _chunk_count(s) -> int:
            try:
                return len(chunker.chunk(s.prompt_body))
            except Exception:
                return 1

        chunk_bins: dict[str, list] = {}
        for s in samples:
            n_chunks = _chunk_count(s)
            if n_chunks <= 3:
                key = "1_3_chunks"
            elif n_chunks <= 6:
                key = "4_6_chunks"
            elif n_chunks <= 12:
                key = "7_12_chunks"
            else:
                key = "13+_chunks"
            chunk_bins.setdefault(key, []).append(s)

        chunk_curve = []
        for bin_name, bin_samples in sorted(chunk_bins.items()):
            if len(bin_samples) < 3:
                continue
            probe = bin_samples[:min(20, len(bin_samples))]
            latencies = _measure_latencies_ms(detector, probe)
            if latencies:
                chunk_curve.append({
                    "bin": bin_name,
                    "n_samples": len(probe),
                    "p50_ms": _pct(latencies, 50),
                    "p95_ms": _pct(latencies, 95),
                })
                click.echo(
                    f"[scaling]     {bin_name}: p50={_pct(latencies,50):.2f}ms  (n={len(probe)})"
                )

        result["latency_vs_chunks_per_prompt"] = chunk_curve

    # ---------------------------------------------------------------
    # 4. Latency vs pair length (intent_content_consistency only)
    # ---------------------------------------------------------------
    if technique == "intent_content_consistency":
        click.echo(f"[scaling]   Measuring latency vs pair length ...")

        def _pair_tokens(s) -> int:
            return (
                len(s.prompt_body.split()) + len(s.advertised_description.split())
            )

        pair_bins = {
            "short_0_256": [s for s in samples if _pair_tokens(s) <= 256],
            "medium_257_512": [s for s in samples if 257 <= _pair_tokens(s) <= 512],
            "long_513_1024": [s for s in samples if 513 <= _pair_tokens(s) <= 1024],
            "very_long_1024+": [s for s in samples if _pair_tokens(s) > 1024],
        }

        pair_curve = []
        for bin_name, bin_samples in pair_bins.items():
            if len(bin_samples) < 3:
                continue
            # Only samples with non-empty descriptions
            valid = [s for s in bin_samples if s.advertised_description.strip()]
            if len(valid) < 3:
                continue
            probe = valid[:min(20, len(valid))]
            latencies = _measure_latencies_ms(detector, probe)
            if latencies:
                pair_curve.append({
                    "bin": bin_name,
                    "n_samples": len(probe),
                    "avg_pair_tokens": float(
                        sum(_pair_tokens(s) for s in probe) / len(probe)
                    ),
                    "p50_ms": _pct(latencies, 50),
                    "p95_ms": _pct(latencies, 95),
                })
                click.echo(
                    f"[scaling]     {bin_name}: p50={_pct(latencies,50):.2f}ms  (n={len(probe)})"
                )

        result["latency_vs_pair_length"] = pair_curve

    return result


def _tokens(sample) -> int:
    """Return token count for a BenchmarkSample."""
    meta = getattr(sample, "metadata", {}) or {}
    if "tokens" in meta:
        return int(meta["tokens"])
    return len(getattr(sample, "prompt_body", "").split())


def _measure_latencies_ms(detector, samples: list) -> list[float]:
    """Measure score_single latency in milliseconds for each sample."""
    import time

    latencies: list[float] = []
    for s in samples:
        try:
            t0 = time.perf_counter()
            detector.score_single(s)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass
    return sorted(latencies)


def _pct(arr: list[float], pct: float) -> float:
    if not arr:
        return 0.0
    idx = min(int(pct / 100.0 * len(arr)), len(arr) - 1)
    return arr[idx]


def _print_scaling_summary(technique: str, result: dict) -> None:
    """Print a compact summary of scaling results."""
    click.echo(f"\n[scaling] Summary for {technique}:")

    for curve_key, curve_data in result.items():
        if not curve_data:
            continue
        click.echo(f"  {curve_key}:")
        for point in curve_data:
            parts = []
            for k, v in point.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.2f}")
                else:
                    parts.append(f"{k}={v}")
            click.echo(f"    {', '.join(parts)}")
