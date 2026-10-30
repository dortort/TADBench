"""Auditable per-example results and input fingerprints for benchmark runs."""

import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None if math.isnan(value) else str(float(value))
    return value


def write_run_artifacts(output_dir, samples, scores, inputs, config, detector, metadata=None):
    """Persist raw scores, class means, coverage diagnostics, and exact input hashes."""
    path = Path(output_dir) / "runs" / uuid4().hex
    path.mkdir(parents=True)
    if len(samples) != len(scores):
        raise ValueError("Every benchmark sample must have one score")
    predictions = []
    for i, (sample, score) in enumerate(zip(samples, scores)):
        row = {"id": sample.id, "label": sample.label, "attack_type": sample.attack_type,
               "source": sample.source, "score": float(score)}
        if metadata is not None:
            row["cascade"] = asdict(metadata[i])
        predictions.append(row)
    diagnostics = {
        "n_samples": len(samples),
        "n_benign": sum(s.label == "benign" for s in samples),
        "n_malicious": sum(s.label == "malicious" for s in samples),
        "mean_score_by_label": {
            label: float(np.mean([score for s, score in zip(samples, scores)
                                  if s.label == label and np.isfinite(score)]))
            if any(s.label == label and np.isfinite(score) for s, score in zip(samples, scores))
            else None for label in ("benign", "malicious")
        },
    }
    from tadbench.detectors.intra_prompt_outlier import IntraPromptOutlierDetector, _pad_chunks
    if isinstance(detector, IntraPromptOutlierDetector):
        from tadbench.evaluation.metrics import compute_all_metrics
        eligible = [len(_pad_chunks(detector._chunker.chunk(s.prompt_body),
                                    detector._fallback_chunker, detector.min_chunks))
                    >= detector.min_chunks for s in samples]
        diagnostics["chunk_eligible_by_label"] = {
            label: sum(ok and s.label == label for s, ok in zip(samples, eligible))
            for label in ("benign", "malicious")
        }
        diagnostics["chunk_eligibility_baseline"] = compute_all_metrics(
            np.array([s.label == "malicious" for s in samples]), np.array(eligible, dtype=float))
        for row, ok in zip(predictions, eligible):
            row["chunk_eligible"] = ok
    from tadbench.detectors.intent_content import IntentContentDetector
    if isinstance(detector, IntentContentDetector):
        diagnostics["body_strategy_changed_samples"] = sum(
            detector._get_body_text(s) != s.prompt_body for s in samples)
        diagnostics["body_strategy_shortened_samples"] = sum(
            len(detector._get_body_text(s).split()) < len(s.prompt_body.split()) for s in samples)
    fingerprints = {}
    for name, source in inputs.items():
        if source is not None:
            source = Path(source)
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
                if digest is None:  # Python 3.10 compatibility
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(65536), b""):
                        digest.update(chunk)
            fingerprints[name] = {"path": str(source), "sha256": digest.hexdigest()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, check=False)
    manifest = {"code_revision": revision.stdout.strip() or None,
                "code_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                "inputs": fingerprints, "config": config, "diagnostics": diagnostics,
                "packages": {d.metadata["Name"]: d.version for d in distributions()}}
    with (path / "predictions.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
    (path / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False))
    return str(path)
