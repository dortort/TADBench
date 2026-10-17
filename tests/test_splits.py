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
