"""Release gates must reject invalid or contaminated published inputs."""

import io
import json
import sys

import pytest

from tadbench.data.release import (
    CONFIGS,
    PREPARATION,
    audit,
    build_release,
    export,
    main,
    sha256,
    verify_release,
)
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
    assert manifest["preparation"] is None
    assert manifest["raw_files"]["input.parquet"] == sha256(raw / "input.parquet")
    assert (destination / "data/threat_bank.jsonl").read_bytes() == (processed / "threat_bank.jsonl").read_bytes()
    card = (destination / "README.md").read_text()
    assert '"example/TADBench"' in card and 'revision="v0.2.0"' in card
    assert "license_link: https://huggingface.co/datasets/example/TADBench/blob/v0.2.0/NOTICE.md" in card
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


def test_verify_does_not_reuse_other_release_cache(processed, tmp_path, monkeypatch):
    monkeypatch.setattr("tadbench.data.release.urlopen", lambda *a, **kw: io.BytesIO(b"Source notice"))
    raw = tmp_path / "raw"
    raw.mkdir()
    first = export(processed, raw, tmp_path / "first/dataset", "v0.1.0")
    assert verify_release(first)["general"]["validation"] == 1
    path = processed / "benchmark_val_combined.jsonl"
    row = json.loads(path.read_text())
    write_jsonl([row, {**row, "id": "extra", "prompt_body": "extra validation input"}], path)
    second = export(processed, raw, tmp_path / "second/dataset", "v0.1.0")
    assert verify_release(second)["general"]["validation"] == 2


def test_build_release_runs_pipeline_in_order_with_recorded_preparation(tmp_path, monkeypatch):
    calls = []

    def record(name):
        def stub(*args, **kwargs):
            calls.append((name, args, kwargs))
            return tmp_path / "dataset" if name == "export" else None
        return stub

    for name, target in (
        ("download", "tadbench.data.download.download_all"),
        ("transform", "tadbench.data.transform.transform_all"),
        ("synthetic", "tadbench.data.generate_synthetic.generate_synthetic"),
        ("adversarial", "tadbench.data.generate_adversarial.generate_adversarial"),
        ("splits", "tadbench.data.splits.build_splits"),
        ("export", "tadbench.data.release.export"),
    ):
        monkeypatch.setattr(target, record(name))
    output = tmp_path / "build"
    assert build_release(output, "v0.3.0", "example/TADBench") == tmp_path / "dataset"
    assert [name for name, *_ in calls] == [
        "download", "transform", "synthetic", "adversarial", "splits", "export"]
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    raw, processed = output / "raw", output / "processed"
    assert by_name["download"][0] == (raw,)
    assert by_name["transform"][0] == (raw, processed)
    assert by_name["synthetic"] == ((processed / "synthetic_mismatch.jsonl",), {
        "n_benign": PREPARATION["synthetic"]["benign"],
        "n_malicious": PREPARATION["synthetic"]["malicious"],
        "n_hard_negatives": PREPARATION["synthetic"]["hard_negatives"],
        "seed": PREPARATION["seed"]})
    per_technique = PREPARATION["adversarial_per_technique"]
    assert by_name["adversarial"] == ((processed,), {
        "n_tps": per_technique, "n_ipsod": per_technique, "n_iccd": per_technique,
        "seed": PREPARATION["seed"]})
    assert by_name["splits"][0] == (processed,)
    assert by_name["export"] == (
        (processed, raw, output / "dataset", "v0.3.0", "example/TADBench"),
        {"preparation": PREPARATION})


def test_build_release_rejects_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        build_release(tmp_path, "v0.1.0")


def test_main_verifies_existing_export(tmp_path, monkeypatch, capsys):
    verified = []
    monkeypatch.setattr("tadbench.data.release.verify_release",
                        lambda path: verified.append(path) or {"general": {"test": 1}})
    monkeypatch.setattr(sys, "argv", ["release", "--verify", str(tmp_path)])
    main()
    assert verified == [tmp_path]
    assert json.loads(capsys.readouterr().out) == {"general": {"test": 1}}


def test_main_builds_then_verifies_release(tmp_path, monkeypatch, capsys):
    built, verified = [], []
    destination = tmp_path / "dataset"
    monkeypatch.setattr("tadbench.data.release.build_release",
                        lambda *args: built.append(args) or destination)
    monkeypatch.setattr("tadbench.data.release.verify_release", verified.append)
    monkeypatch.setattr(sys, "argv", ["release", "--output-dir", str(tmp_path / "build"),
                                      "--version", "v0.3.0", "--repo-id", "example/TADBench"])
    main()
    assert built == [(tmp_path / "build", "v0.3.0", "example/TADBench")]
    assert verified == [destination]
    assert str(destination) in capsys.readouterr().out
