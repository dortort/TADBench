"""Tests for tadbench.data.splits."""

from __future__ import annotations

from pathlib import Path

from tadbench.data.splits import build_combined_sets
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
