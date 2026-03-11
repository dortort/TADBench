"""Technique/dataset compatibility helpers for benchmark execution."""

from __future__ import annotations

from pathlib import Path

from tadbench.data.schema import BenchmarkSample, has_description

DESCRIPTION_REQUIRED_TECHNIQUES = {
    "intent_content_consistency",
    "cascade",
}

DESCRIPTION_COMPATIBLE_DATASET = "benchmark_description_combined.jsonl"
DESCRIPTION_COMPATIBLE_VAL_DATASET = "benchmark_description_val_combined.jsonl"


def technique_requires_descriptions(technique: str) -> bool:
    """Return True when a technique requires every sample to have a description."""
    return technique in DESCRIPTION_REQUIRED_TECHNIQUES


def validate_dataset_for_technique(
    technique: str,
    samples: list[BenchmarkSample],
    dataset_path: str | Path,
) -> None:
    """Raise ValueError when *samples* are incompatible with *technique*."""
    if not technique_requires_descriptions(technique):
        return

    missing = sum(1 for sample in samples if not has_description(sample))
    if missing == 0:
        return

    dataset_name = Path(dataset_path).name
    raise ValueError(
        f"Technique '{technique}' requires advertised_description for every sample, "
        f"but {missing}/{len(samples)} samples in '{dataset_name}' are missing it. "
        f"Use '{DESCRIPTION_COMPATIBLE_DATASET}' for evaluation or "
        f"'{DESCRIPTION_COMPATIBLE_VAL_DATASET}' for calibration."
    )
