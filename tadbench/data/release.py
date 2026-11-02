"""Build an auditable Hugging Face dataset export without publishing it.

Run: python -m tadbench.data.release --output-dir data/releases/v0.1.0
The destination must not exist: every release starts with fresh preparation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from tadbench.config import HF_DATASETS, PROJECT_ROOT
from tadbench.data.schema import has_description, validate_sample
from tadbench.data.validation import load_threat_bank, validate_disjoint_samples
from tadbench.utils.io import read_jsonl


CONFIGS = {
    "general": {
        "validation": ["benchmark_val_combined.jsonl"],
        "test": ["benchmark_combined.jsonl"],
    },
    "description": {
        "validation": ["benchmark_description_val_combined.jsonl"],
        "test": ["benchmark_description_combined.jsonl"],
    },
    **{source: {split: [f"{source}_{suffix}.jsonl"]
                for split, suffix in (("train", "train"), ("validation", "val"), ("test", "test"))}
       for source in ("promptshield", "deepset", "spml", "synthetic_mismatch")},
    "adversarial": {"test": [f"adversarial_{kind}_evasion.jsonl"
                               for kind in ("tps", "ipsod", "iccd")]},
    "threat_bank": {"train": ["threat_bank.jsonl"]},
}

# Explicit nullable metadata fields prevent incompatible Arrow inference across
# files (for example, TPS evasion has evasion_technique, IPSOD has n_sentences).
METADATA_TYPES = {
    "tokens": "int64", "language": "string", "severity": "int64",
    "category": "string", "evasion_technique": "string", "n_sentences": "int64",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(processed: Path) -> dict:
    """Reject invalid schemas, contaminated calibration/test data, or banks."""
    statistics = {}
    for path in sorted(processed.glob("*.jsonl")):
        samples = [validate_sample(row) for row in read_jsonl(path)]
        for sample in samples:
            unknown = set(sample.metadata) - METADATA_TYPES.keys()
            if unknown:
                raise ValueError(f"Unmapped release metadata fields: {sorted(unknown)}")
        statistics[path.name] = {
            "rows": len(samples),
            "labels": dict(Counter(sample.label for sample in samples)),
            "sources": dict(Counter(sample.source for sample in samples)),
        }
    for config in CONFIGS.values():
        for filenames in config.values():
            for filename in filenames:
                if filename not in statistics or not statistics[filename]["rows"]:
                    raise ValueError(f"Required release input is missing or empty: {filename}")
    for name in ("general", "description"):
        config = CONFIGS[name]
        validation = [validate_sample(row) for row in read_jsonl(processed / config["validation"][0])]
        test = [validate_sample(row) for row in read_jsonl(processed / config["test"][0])]
        validate_disjoint_samples(validation, test)
        load_threat_bank(processed / "threat_bank.jsonl", validation + test)
        if name == "description" and not all(has_description(s) for s in validation + test):
            raise ValueError("Description-compatible data contains missing descriptions")
    training_rows = {
        json.dumps(row, sort_keys=True) for source in HF_DATASETS
        for row in read_jsonl(processed / f"{source}_train.jsonl")
        if row["label"] == "malicious"
    }
    if any(json.dumps(row, sort_keys=True) not in training_rows
           for row in read_jsonl(processed / "threat_bank.jsonl")):
        raise ValueError("Threat bank contains samples outside malicious source training splits")
    return statistics


def card_header(repo_id: str = "dortort/TADBench") -> str:
    """Use explicit configs so archival/auxiliary JSONL files are never inferred."""
    lines = ["---", "pretty_name: TADBench", "language:", "- en",
             "license: other", "license_name: source-specific-licenses",
             f"license_link: https://huggingface.co/datasets/{repo_id}/blob/main/NOTICE.md",
             "task_categories:", "- text-classification",
             "tags:", "- prompt-injection", "- benchmark", "- llm-security", "configs:"]
    for name, config in CONFIGS.items():
        lines.append(f"- config_name: {name}")
        if name == "general":
            lines.append("  default: true")
        lines.append("  data_files:")
        for split, filenames in config.items():
            lines.extend([f"  - split: {split}", "    path:"])
            lines.extend(f"    - data/{filename}" for filename in filenames)
    lines.append("dataset_info:")
    for name in CONFIGS:
        lines.extend([f"- config_name: {name}", "  features:"])
        for field in ("id", "advertised_description", "prompt_body", "label", "attack_type", "source"):
            lines.extend([f"  - name: {field}", "    dtype: string"])
        lines.extend(["  - name: metadata", "    struct:"])
        for field, dtype in METADATA_TYPES.items():
            lines.extend([f"    - name: {field}", f"      dtype: {dtype}"])
    return "\n".join(lines) + "\n---\n\n"


def export(processed: Path, raw: Path, destination: Path, version: str,
           repo_id: str = "dortort/TADBench") -> Path:
    """Export already prepared inputs; the caller must establish their provenance."""
    stats = audit(processed)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(processed, destination / "data")
    provenance = destination / "provenance"
    provenance.mkdir()
    for name, config in HF_DATASETS.items():
        url = f"https://huggingface.co/datasets/{config['path']}/resolve/{config['revision']}/README.md"
        with urlopen(url, timeout=60) as response:
            (provenance / f"{name}-README.md").write_bytes(response.read())
    licenses = destination / "licenses"
    licenses.mkdir()
    for filename, url in {
        "Apache-2.0.txt": "https://raw.githubusercontent.com/spdx/license-list-data/main/text/Apache-2.0.txt",
        "CC-BY-4.0.txt": "https://raw.githubusercontent.com/spdx/license-list-data/main/text/CC-BY-4.0.txt",
    }.items():
        with urlopen(url, timeout=60) as response:
            (licenses / filename).write_bytes(response.read())
    docs = PROJECT_ROOT / "docs" / "huggingface"
    card = (docs / "DATASET_CARD.md").read_text().replace("YOUR_NAMESPACE/TADBench", repo_id)
    (destination / "README.md").write_text(card_header(repo_id) + card.replace("v0.1.0", version))
    shutil.copyfile(docs / "NOTICE.md", destination / "NOTICE.md")
    shutil.copyfile(PROJECT_ROOT / "LICENSE", destination / "TADBench-LICENSE.txt")
    shutil.copyfile(PROJECT_ROOT / "matrix_config.json", provenance / "matrix_config.json")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()
    sources = list((PROJECT_ROOT / "tadbench").rglob("*.py")) + [
        PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "matrix_config.json", PROJECT_ROOT / "LICENSE",
        docs / "DATASET_CARD.md", docs / "NOTICE.md",
    ]
    for path in sources:
        target = provenance / "source" / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = {
        "version": version, "repo_id": repo_id,
        "code_revision": revision, "code_dirty": bool(dirty),
        "upstream_datasets": HF_DATASETS,
        "preparation": {"seed": 42, "synthetic": {"benign": 1000, "malicious": 1000, "hard_negatives": 1000},
                        "adversarial_per_technique": 200,
                        "ids": "UUID4; use the archived JSONL for exact IDs and bytes"},
        "checks": {"schemas": "passed", "validation_test_disjoint": "passed",
                   "threat_bank_disjoint": "passed", "threat_bank_training_membership": "passed"},
        "statistics": stats,
        "raw_files": {str(path.relative_to(raw)): sha256(path) for path in sorted(raw.rglob("*.parquet"))},
        "source_files": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sorted(sources)},
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "files": {str(path.relative_to(destination)): sha256(path)
                  for path in sorted(destination.rglob("*")) if path.is_file()},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def verify_release(destination: Path) -> dict:
    """Check archived bytes and load every config using only the local export."""
    from datasets import load_dataset

    manifest = json.loads((destination / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        if sha256(destination / filename) != digest:
            raise ValueError(f"Release checksum mismatch: {filename}")
    counts = {}
    # Local directories with the same basename/card can collide in the global
    # datasets cache. Verify the actual export in a fresh, disposable cache.
    with TemporaryDirectory(prefix="tadbench-verify-") as cache:
        for name, config in CONFIGS.items():
            dataset = load_dataset(str(destination.resolve()), name=name, cache_dir=cache)
            if set(dataset) != set(config):
                raise ValueError(f"Unexpected splits in configuration {name}")
            counts[name] = {}
            for split, filenames in config.items():
                expected = sum(manifest["statistics"][filename]["rows"] for filename in filenames)
                if len(dataset[split]) != expected:
                    raise ValueError(f"Unexpected row count in {name}/{split}")
                counts[name][split] = expected
    return counts


def build_release(output: Path, version: str, repo_id: str = "dortort/TADBench") -> Path:
    from tadbench.data.download import download_all
    from tadbench.data.generate_adversarial import generate_adversarial
    from tadbench.data.generate_synthetic import generate_synthetic
    from tadbench.data.splits import build_splits
    from tadbench.data.transform import transform_all

    output.mkdir(parents=True, exist_ok=False)
    raw, processed = output / "raw", output / "processed"
    download_all(raw)
    transform_all(raw, processed)
    generate_synthetic(processed / "synthetic_mismatch.jsonl", seed=42)
    generate_adversarial(processed, seed=42)
    build_splits(processed)
    return export(processed, raw, output / "dataset", version, repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--verify", type=Path, help="Verify an existing dataset/ export without rebuilding")
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument("--repo-id", default="dortort/TADBench")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_release(args.verify), indent=2))
        return
    destination = build_release(args.output_dir, args.version, args.repo_id)
    verify_release(destination)
    print(f"Audited dataset export ready: {destination}")


if __name__ == "__main__":
    main()
