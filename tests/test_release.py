"""Release gates must reject invalid or contaminated published inputs."""

import io
import json

import pytest

from tadbench.data.release import CONFIGS, audit, export, sha256, verify_release
from tadbench.utils.io import write_jsonl


@pytest.fixture
def processed(tmp_path):
    path = tmp_path / "processed"
    for config in CONFIGS.values():
        for filenames in config.values():
            for filename in filenames:
                source = next((s for s in ("promptshield", "deepset", "spml", "synthetic_mismatch")
                               if filename.startswith(s)), "adversarial")
                row = {"id": filename, "advertised_description": "task",
                       "prompt_body": filename, "label": "malicious",
                       "attack_type": "direct_injection", "source": source,
                       "metadata": {}}
                write_jsonl([row], path / filename)
    (path / "threat_bank.jsonl").write_bytes((path / "promptshield_train.jsonl").read_bytes())
    return path


def test_audit_accepts_disjoint_inputs(processed):
    assert audit(processed)["threat_bank.jsonl"]["rows"] == 1


def test_audit_rejects_calibration_test_overlap(processed):
    (processed / "benchmark_combined.jsonl").write_bytes(
        (processed / "benchmark_val_combined.jsonl").read_bytes())
    with pytest.raises(ValueError, match="Validation and test inputs overlap"):
        audit(processed)


def test_audit_rejects_bank_overlap(processed):
    (processed / "benchmark_combined.jsonl").write_bytes(
        (processed / "threat_bank.jsonl").read_bytes())
    with pytest.raises(ValueError, match="Threat bank overlaps"):
        audit(processed)


def test_audit_rejects_missing_inputs(processed):
    (processed / "spml_test.jsonl").unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        audit(processed)


def test_audit_rejects_bank_outside_training(processed):
    (processed / "threat_bank.jsonl").write_bytes(
        (processed / "adversarial_tps_evasion.jsonl").read_bytes())
    with pytest.raises(ValueError, match="outside malicious source training"):
        audit(processed)


def test_audit_rejects_changed_training_row_in_bank(processed):
    row = json.loads((processed / "threat_bank.jsonl").read_text())
    row["prompt_body"] = "Modified training input with the same ID"
    write_jsonl([row], processed / "threat_bank.jsonl")
    with pytest.raises(ValueError, match="outside malicious source training"):
        audit(processed)


def test_export_preserves_bytes_and_fingerprints(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("tadbench.data.release.urlopen", lambda *a, **kw: io.BytesIO(b"Source notice"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "input.parquet").write_bytes(b"raw fixture")
    destination = export(processed, raw, tmp_path / "dataset", "v0.2.0", "example/TADBench")
    manifest = json.loads((destination / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        assert sha256(destination / filename) == digest
    assert manifest["raw_files"]["input.parquet"] == sha256(raw / "input.parquet")
    assert (destination / "data/threat_bank.jsonl").read_bytes() == (processed / "threat_bank.jsonl").read_bytes()
    card = (destination / "README.md").read_text()
    assert '"example/TADBench"' in card and 'revision="v0.2.0"' in card
    assert "config_name: description" in card
    with pytest.raises(FileExistsError):
        export(processed, raw, destination, "v0.2.0")
    (destination / "data/threat_bank.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release(destination)


def test_all_configs_load_with_different_metadata_fields(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("tadbench.data.release.urlopen", lambda *a, **kw: io.BytesIO(b"Source notice"))
    for filename, metadata in (
        ("adversarial_tps_evasion.jsonl", {"evasion_technique": "test"}),
        ("adversarial_ipsod_evasion.jsonl", {"n_sentences": 8}),
    ):
        row = json.loads((processed / filename).read_text())
        row["metadata"] = metadata
        write_jsonl([row], processed / filename)
    raw = tmp_path / "raw"
    raw.mkdir()
    destination = export(processed, raw, tmp_path / "dataset", "v0.1.0")
    counts = verify_release(destination)
    for name, config in CONFIGS.items():
        assert set(counts[name]) == set(config)
        for split, filenames in config.items():
            assert counts[name][split] == len(filenames)
