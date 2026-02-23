"""Build train/val/test splits, threat bank, length-binned subsets, and combined sets."""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tadbench.config import PROCESSED_DIR
from tadbench.utils.io import read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LENGTH_BINS: dict[str, tuple[int, int]] = {
    "short": (0, 128),
    "medium": (129, 512),
    "long": (513, 2048),
    "very_long": (2049, 10**9),
}

THREAT_BANK_SOURCES = ["promptshield", "deepset", "spml"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_tokens(sample: dict[str, Any]) -> int:
    """Return the token count for a sample (falls back to body split length)."""
    meta = sample.get("metadata", {})
    if "tokens" in meta:
        return int(meta["tokens"])
    return len(sample.get("prompt_body", "").split())


def _stratified_split(
    samples: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by label.

    Returns (train, val, test) lists.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    train_all: list[dict] = []
    val_all: list[dict] = []
    test_all: list[dict] = []

    for label, group in by_label.items():
        g = group[:]
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_all.extend(g[:n_test])
        val_all.extend(g[n_test: n_test + n_val])
        train_all.extend(g[n_test + n_val:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)
    return train_all, val_all, test_all


def _write_if_nonempty(records: list[dict], path: Path, label: str) -> None:
    if records:
        write_jsonl(records, path)
        print(f"[splits]   {label}: {len(records):,} samples -> {path}")
    else:
        print(f"[splits]   {label}: (empty, skipped)")


# ---------------------------------------------------------------------------
# Threat bank
# ---------------------------------------------------------------------------

def build_threat_bank(
    processed_dir: Path,
    sources: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract malicious prompts from train splits and write threat_bank.jsonl.

    Args:
        processed_dir: Directory containing processed JSONL files.
        sources: Dataset names to include (default: all three).
        output_path: Where to write the threat bank.

    Returns:
        List of malicious sample dicts.
    """
    if sources is None:
        sources = THREAT_BANK_SOURCES
    out = output_path or (processed_dir / "threat_bank.jsonl")

    threat_bank: list[dict[str, Any]] = []
    for source in sources:
        train_path = processed_dir / f"{source}_train.jsonl"
        if not train_path.exists():
            print(f"[splits] Threat bank: {train_path} not found, skipping")
            continue
        records = read_jsonl(train_path)
        malicious = [r for r in records if r.get("label") == "malicious"]
        threat_bank.extend(malicious)
        print(
            f"[splits] Threat bank: {source}_train -> "
            f"{len(malicious):,}/{len(records):,} malicious"
        )

    write_jsonl(threat_bank, out)
    print(f"[splits] Threat bank: {len(threat_bank):,} total -> {out}")
    return threat_bank


# ---------------------------------------------------------------------------
# Dataset-specific split logic
# ---------------------------------------------------------------------------

def _split_promptshield(processed_dir: Path) -> None:
    """PromptShield uses existing train/val/test splits from HuggingFace."""
    print("[splits] PromptShield: using existing splits (train/val/test from HF)")
    for split in ("train", "val", "test"):
        path = processed_dir / f"promptshield_{split}.jsonl"
        if path.exists():
            records = read_jsonl(path)
            print(f"[splits]   promptshield_{split}: {len(records):,} samples")
        else:
            print(f"[splits]   promptshield_{split}: not found")


def _split_deepset(processed_dir: Path, val_frac: float = 0.2, seed: int = 42) -> None:
    """deepset: use existing test split; create val by holding 20% from train."""
    print("[splits] deepset: creating val from 20% of train ...")
    train_path = processed_dir / "deepset_train.jsonl"
    if not train_path.exists():
        print("[splits]   deepset_train.jsonl not found, skipping")
        return

    all_train = read_jsonl(train_path)
    # Stratified split: (1 - val_frac) -> train, val_frac -> val
    train_out, val_out, _ = _stratified_split(
        all_train, val_frac=val_frac, test_frac=0.0, seed=seed
    )
    # test_frac=0.0 puts nothing in test; we use existing deepset_test.jsonl

    _write_if_nonempty(train_out, processed_dir / "deepset_train.jsonl", "deepset_train (updated)")
    _write_if_nonempty(val_out, processed_dir / "deepset_val.jsonl", "deepset_val")

    test_path = processed_dir / "deepset_test.jsonl"
    if test_path.exists():
        test_records = read_jsonl(test_path)
        print(f"[splits]   deepset_test: {len(test_records):,} samples (existing)")


def _split_spml(
    processed_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """SPML: stratified 70/15/15 split."""
    print("[splits] SPML: creating stratified 70/15/15 split ...")
    # SPML has one large split (train); we re-split it entirely
    source_path = processed_dir / "spml_train.jsonl"
    if not source_path.exists():
        print("[splits]   spml_train.jsonl not found, skipping")
        return

    all_records = read_jsonl(source_path)
    test_frac = 1.0 - train_frac - val_frac
    train_out, val_out, test_out = _stratified_split(
        all_records, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    _write_if_nonempty(train_out, processed_dir / "spml_train.jsonl", "spml_train")
    _write_if_nonempty(val_out, processed_dir / "spml_val.jsonl", "spml_val")
    _write_if_nonempty(test_out, processed_dir / "spml_test.jsonl", "spml_test")


def _split_synthetic(
    processed_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """Synthetic mismatch: stratified 70/15/15 split."""
    print("[splits] Synthetic mismatch: creating stratified 70/15/15 split ...")
    source_path = processed_dir / "synthetic_mismatch.jsonl"
    if not source_path.exists():
        print("[splits]   synthetic_mismatch.jsonl not found, skipping")
        return

    all_records = read_jsonl(source_path)
    test_frac = 1.0 - train_frac - val_frac
    train_out, val_out, test_out = _stratified_split(
        all_records, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    _write_if_nonempty(train_out, processed_dir / "synthetic_mismatch_train.jsonl", "synthetic_train")
    _write_if_nonempty(val_out, processed_dir / "synthetic_mismatch_val.jsonl", "synthetic_val")
    _write_if_nonempty(test_out, processed_dir / "synthetic_mismatch_test.jsonl", "synthetic_test")


# ---------------------------------------------------------------------------
# Length-binned subsets
# ---------------------------------------------------------------------------

def build_length_bins(
    test_files: list[Path],
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Create length-binned subsets of the combined test data.

    Args:
        test_files: List of test JSONL file paths to read.
        output_dir: Directory to write binned JSONL files.

    Returns:
        Dict mapping bin name -> list of sample dicts.
    """
    print("[splits] Building length-binned subsets ...")
    all_test: list[dict[str, Any]] = []
    for p in test_files:
        if p.exists():
            all_test.extend(read_jsonl(p))

    bins: dict[str, list[dict[str, Any]]] = {name: [] for name in LENGTH_BINS}
    for sample in all_test:
        tokens = _get_tokens(sample)
        for bin_name, (lo, hi) in LENGTH_BINS.items():
            if lo <= tokens <= hi:
                bins[bin_name].append(sample)
                break

    for bin_name, records in bins.items():
        if records:
            out_path = output_dir / f"benchmark_test_{bin_name}.jsonl"
            write_jsonl(records, out_path)
            print(f"[splits]   {bin_name} (tokens {LENGTH_BINS[bin_name]}): {len(records):,} -> {out_path}")

    return bins


# ---------------------------------------------------------------------------
# Drift holdout splits
# ---------------------------------------------------------------------------

def build_drift_holdout_splits(
    all_samples: list[dict[str, Any]],
    output_dir: Path,
    attack_types: list[str] | None = None,
) -> None:
    """Write per-attack-type holdout files for drift evaluation.

    For each attack_type, writes a file containing only those samples.
    """
    if attack_types is None:
        # Discover attack types present in the data
        attack_types = sorted({s["attack_type"] for s in all_samples if s["attack_type"] != "none"})

    for att in attack_types:
        holdout = [s for s in all_samples if s["attack_type"] == att]
        if len(holdout) < 10:
            print(f"[splits] Drift holdout '{att}': only {len(holdout)} samples (< 10), skipping")
            continue
        out_path = output_dir / f"drift_holdout_{att}.jsonl"
        write_jsonl(holdout, out_path)
        print(f"[splits] Drift holdout '{att}': {len(holdout):,} -> {out_path}")


# ---------------------------------------------------------------------------
# Combined sets
# ---------------------------------------------------------------------------

def build_combined_sets(processed_dir: Path) -> None:
    """Build benchmark_combined.jsonl and benchmark_val_combined.jsonl."""
    print("[splits] Building combined benchmark sets ...")

    test_sources = [
        "promptshield_test",
        "deepset_test",
        "spml_test",
        "synthetic_mismatch_test",
        "adversarial_tps_evasion",
        "adversarial_ipsod_evasion",
        "adversarial_iccd_evasion",
    ]
    val_sources = [
        "promptshield_val",
        "deepset_val",
        "spml_val",
        "synthetic_mismatch_val",
    ]

    combined_test: list[dict[str, Any]] = []
    for name in test_sources:
        p = processed_dir / f"{name}.jsonl"
        if p.exists():
            records = read_jsonl(p)
            combined_test.extend(records)
            print(f"[splits]   +{name}: {len(records):,}")

    combined_val: list[dict[str, Any]] = []
    for name in val_sources:
        p = processed_dir / f"{name}.jsonl"
        if p.exists():
            records = read_jsonl(p)
            combined_val.extend(records)
            print(f"[splits]   +{name} (val): {len(records):,}")

    if combined_test:
        out_path = processed_dir / "benchmark_combined.jsonl"
        write_jsonl(combined_test, out_path)
        malicious = sum(1 for r in combined_test if r.get("label") == "malicious")
        print(
            f"[splits] benchmark_combined.jsonl: {len(combined_test):,} total, "
            f"{malicious:,} malicious -> {out_path}"
        )

    if combined_val:
        out_path = processed_dir / "benchmark_val_combined.jsonl"
        write_jsonl(combined_val, out_path)
        malicious = sum(1 for r in combined_val if r.get("label") == "malicious")
        print(
            f"[splits] benchmark_val_combined.jsonl: {len(combined_val):,} total, "
            f"{malicious:,} malicious -> {out_path}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_splits(
    processed_dir: Path | str | None = None,
    threat_bank_sources: list[str] | None = None,
) -> None:
    """Run all splitting steps.

    1. Build threat bank from train splits.
    2. Apply per-dataset split logic.
    3. Build length-binned test subsets.
    4. Build drift holdout files.
    5. Build combined benchmark sets.

    Args:
        processed_dir: Directory containing processed JSONL files.
                       Defaults to ``tadbench.config.PROCESSED_DIR``.
        threat_bank_sources: Dataset names for threat bank.
                             Defaults to all three.
    """
    out = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[splits] Step 1: Threat bank")
    print("=" * 60)
    build_threat_bank(out, sources=threat_bank_sources)

    print("=" * 60)
    print("[splits] Step 2: Dataset-specific splits")
    print("=" * 60)
    _split_promptshield(out)
    _split_deepset(out)
    _split_spml(out)
    _split_synthetic(out)

    print("=" * 60)
    print("[splits] Step 3: Length-binned subsets")
    print("=" * 60)
    test_files = [
        out / "promptshield_test.jsonl",
        out / "deepset_test.jsonl",
        out / "spml_test.jsonl",
        out / "synthetic_mismatch_test.jsonl",
    ]
    build_length_bins(test_files, out)

    print("=" * 60)
    print("[splits] Step 4: Drift holdout splits")
    print("=" * 60)
    # Load all available data for drift holdout files
    all_for_drift: list[dict[str, Any]] = []
    for fname in out.glob("*.jsonl"):
        if "drift_holdout" not in fname.name and "threat_bank" not in fname.name:
            try:
                all_for_drift.extend(read_jsonl(fname))
            except Exception:
                pass
    build_drift_holdout_splits(all_for_drift, out)

    print("=" * 60)
    print("[splits] Step 5: Combined sets")
    print("=" * 60)
    build_combined_sets(out)

    print(f"\n[splits] Done. All files in: {out}")


if __name__ == "__main__":
    dir_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    build_splits(dir_arg)
