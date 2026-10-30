"""Tests for tadbench.data.splits."""

from __future__ import annotations

from pathlib import Path

from tadbench.data.splits import build_combined_sets, build_splits
from tadbench.utils.io import read_jsonl, write_jsonl


def _record(
    sample_id: str,
    source: str,
    desc: str,
    label: str = "benign",
    attack_type: str = "none",
) -> dict:
    body = f"Prompt body for {sample_id}"
    return {
        "id": sample_id,
        "advertised_description": desc,
        "prompt_body": body,
        "label": label,
        "attack_type": attack_type,
        "source": source,
        "metadata": {"tokens": len(body.split()), "language": "en"},
    }


class TestBuildCombinedSets:
    def test_writes_description_compatible_combined_sets(self, tmp_path: Path):
        write_jsonl(
            [_record("ps-test", "promptshield", desc="")],
            tmp_path / "promptshield_test.jsonl",
        )
        write_jsonl(
            [_record("ds-test", "deepset", desc="")],
            tmp_path / "deepset_test.jsonl",
        )
        write_jsonl(
            [_record("spml-test", "spml", desc="system prompt")],
            tmp_path / "spml_test.jsonl",
        )
        write_jsonl(
            [_record("syn-test", "synthetic_mismatch", desc="task description")],
            tmp_path / "synthetic_mismatch_test.jsonl",
        )
        write_jsonl(
            [_record("adv-test", "adversarial", desc="security review", label="malicious", attack_type="iccd_evasion")],
            tmp_path / "adversarial_iccd_evasion.jsonl",
        )
        write_jsonl(
            [_record("ps-val", "promptshield", desc="")],
            tmp_path / "promptshield_val.jsonl",
        )
        write_jsonl(
            [_record("spml-val", "spml", desc="validation description")],
            tmp_path / "spml_val.jsonl",
        )

        build_combined_sets(tmp_path)

        combined = read_jsonl(tmp_path / "benchmark_combined.jsonl")
        desc_combined = read_jsonl(tmp_path / "benchmark_description_combined.jsonl")
        desc_val = read_jsonl(tmp_path / "benchmark_description_val_combined.jsonl")

        assert len(combined) == 5
        assert len(desc_combined) == 3
        assert all(record["advertised_description"].strip() for record in desc_combined)
        assert len(desc_val) == 1
        assert desc_val[0]["id"] == "spml-val"


class TestBuildSplitsThreatBank:
    def test_threat_bank_excludes_held_out_samples(self, tmp_path: Path):
        spml = [
            _record(f"spml-{i}", "spml", desc="system prompt", label=label,
                    attack_type="direct_injection" if label == "malicious" else "none")
            for i, label in enumerate(["malicious", "benign"] * 20)
        ]
        deepset = [
            _record(f"ds-{i}", "deepset", desc="", label=label,
                    attack_type="direct_injection" if label == "malicious" else "none")
            for i, label in enumerate(["malicious", "benign"] * 10)
        ]
        write_jsonl(spml, tmp_path / "spml_train.jsonl")
        write_jsonl(deepset, tmp_path / "deepset_train.jsonl")

        build_splits(tmp_path, threat_bank_sources=["spml", "deepset"])

        bank_ids = {r["id"] for r in read_jsonl(tmp_path / "threat_bank.jsonl")}
        held_out_ids = {
            r["id"]
            for name in ("spml_val", "spml_test", "deepset_val")
            for r in read_jsonl(tmp_path / f"{name}.jsonl")
        }
        train_malicious_ids = {
            r["id"]
            for name in ("spml_train", "deepset_train")
            for r in read_jsonl(tmp_path / f"{name}.jsonl")
            if r["label"] == "malicious"
        }

        assert held_out_ids, "fixture should produce held-out splits"
        assert bank_ids.isdisjoint(held_out_ids)
        assert bank_ids == train_malicious_ids


def test_zero_test_fraction_preserves_all_rows():
    from tadbench.data.splits import _stratified_split
    records = [_record(str(i), "deepset", "", label=label)
               for i, label in enumerate(["benign", "malicious"] * 10)]
    train, val, test = _stratified_split(records, val_frac=.2, test_frac=0)
    assert not test
    assert len(train) + len(val) == len(records)
    assert {r["id"] for r in train}.isdisjoint(r["id"] for r in val)


def test_drift_holdouts_exclude_train_and_derived_files(tmp_path):
    train = [_record(f"train-{i}", "promptshield", "", "malicious", "direct_injection")
             for i in range(20)]
    test = [_record(f"test-{i}", "promptshield", "", "malicious", "direct_injection")
            for i in range(12)]
    write_jsonl(train, tmp_path / "promptshield_train.jsonl")
    write_jsonl(test, tmp_path / "promptshield_test.jsonl")
    write_jsonl(train + test + test, tmp_path / "benchmark_combined.jsonl")
    build_splits(tmp_path)
    holdout = read_jsonl(tmp_path / "drift_holdout_direct_injection.jsonl")
    assert {r["id"] for r in holdout} == {r["id"] for r in test}
    assert len(holdout) == len(test)


def test_repeated_template_inputs_remain_in_one_split():
    from tadbench.data.splits import _stratified_split
    records = [_record(str(i), "synthetic_mismatch", "describe") for i in range(30)]
    for i, record in enumerate(records):
        record["prompt_body"] = f"template {i % 10}"
    splits = _stratified_split(records, .2, .2)
    inputs = [{r["prompt_body"] for r in split} for split in splits]
    assert sum(map(len, splits)) == len(records)
    assert inputs[0].isdisjoint(inputs[1] | inputs[2])
    assert inputs[1].isdisjoint(inputs[2])


def test_threat_bank_excludes_cross_source_copies(tmp_path):
    from tadbench.data.splits import build_threat_bank
    train = _record("train", "spml", "", "malicious", "direct_injection")
    copied = {**train, "id": "test", "source": "deepset"}
    write_jsonl([train], tmp_path / "spml_train.jsonl")
    write_jsonl([copied], tmp_path / "deepset_test.jsonl")
    assert build_threat_bank(tmp_path, sources=["spml"]) == []
