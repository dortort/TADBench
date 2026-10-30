"""Disjointness checks for benchmark fitting and evaluation inputs."""

from pathlib import Path

from tadbench.data.schema import BenchmarkSample, validate_sample
from tadbench.utils.io import read_jsonl


def load_threat_bank(path: str | Path | None, evaluation: list[BenchmarkSample]) -> list[BenchmarkSample]:
    """Require an explicit malicious-only bank disjoint from evaluation prompts."""
    if path is None or not Path(path).is_file():
        raise ValueError("A training threat bank is required; prepare data or pass threat_bank_path")
    bank = [validate_sample(row) for row in read_jsonl(path)]
    if not bank or any(s.label != "malicious" for s in bank):
        raise ValueError("Threat bank must contain nonempty, malicious-only training data")
    bank_ids = {s.id for s in bank}
    bank_bodies = {s.prompt_body.strip() for s in bank}
    overlap = [s.id for s in evaluation if s.id in bank_ids or s.prompt_body.strip() in bank_bodies]
    if overlap:
        raise ValueError(f"Threat bank overlaps evaluation data ({len(overlap)} samples); rebuild disjoint splits")
    return bank
