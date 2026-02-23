"""Tests for tadbench.data.schema."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from tadbench.data.schema import (
    BenchmarkSample,
    has_description,
    validate_jsonl,
    validate_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_dict(**overrides) -> dict:
    base = {
        "id": str(uuid.uuid4()),
        "advertised_description": "",
        "prompt_body": "Summarize the following text.",
        "label": "benign",
        "attack_type": "none",
        "source": "deepset",
        "metadata": {"tokens": 5, "language": "en"},
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# validate_sample — valid cases
# ---------------------------------------------------------------------------

class TestValidateSampleValid:
    def test_benign_sample_returns_benchmark_sample(self):
        d = _valid_dict()
        s = validate_sample(d)
        assert isinstance(s, BenchmarkSample)
        assert s.label == "benign"
        assert s.attack_type == "none"

    def test_malicious_sample(self):
        d = _valid_dict(label="malicious", attack_type="direct_injection")
        s = validate_sample(d)
        assert s.label == "malicious"
        assert s.attack_type == "direct_injection"

    def test_all_valid_attack_types(self):
        valid_attacks = [
            "none", "direct_injection", "hidden_instruction", "role_override",
            "exfiltration", "jailbreak", "indirect_injection", "tool_misuse",
            "tps_evasion", "ipsod_evasion", "iccd_evasion",
        ]
        for at in valid_attacks:
            lbl = "benign" if at == "none" else "malicious"
            d = _valid_dict(label=lbl, attack_type=at)
            s = validate_sample(d)
            assert s.attack_type == at

    def test_all_valid_sources(self):
        valid_sources = ["promptshield", "deepset", "spml", "synthetic_mismatch", "adversarial"]
        for src in valid_sources:
            d = _valid_dict(source=src)
            s = validate_sample(d)
            assert s.source == src

    def test_empty_metadata_allowed(self):
        d = _valid_dict(metadata={})
        s = validate_sample(d)
        assert s.metadata == {}

    def test_missing_metadata_defaults_to_empty_dict(self):
        d = _valid_dict()
        del d["metadata"]
        s = validate_sample(d)
        assert s.metadata == {}

    def test_with_advertised_description(self):
        d = _valid_dict(advertised_description="Summarize this document.")
        s = validate_sample(d)
        assert s.advertised_description == "Summarize this document."

    def test_autogenerate_id_when_empty(self):
        d = _valid_dict(id="")
        s = validate_sample(d)
        assert s.id != ""
        assert len(s.id) > 0


# ---------------------------------------------------------------------------
# validate_sample — invalid cases
# ---------------------------------------------------------------------------

class TestValidateSampleInvalid:
    def test_invalid_label_raises_value_error(self):
        d = _valid_dict(label="unknown")
        with pytest.raises(ValueError, match="Invalid label"):
            validate_sample(d)

    def test_invalid_attack_type_raises_value_error(self):
        d = _valid_dict(attack_type="mega_hack")
        with pytest.raises(ValueError, match="Invalid attack_type"):
            validate_sample(d)

    def test_invalid_source_raises_value_error(self):
        d = _valid_dict(source="bad_source")
        with pytest.raises(ValueError, match="Invalid source"):
            validate_sample(d)

    def test_missing_prompt_body_raises(self):
        d = _valid_dict()
        del d["prompt_body"]
        with pytest.raises(ValueError, match="Missing required field"):
            validate_sample(d)

    def test_missing_label_raises(self):
        d = _valid_dict()
        del d["label"]
        with pytest.raises(ValueError, match="Missing required field"):
            validate_sample(d)

    def test_missing_attack_type_raises(self):
        d = _valid_dict()
        del d["attack_type"]
        with pytest.raises(ValueError, match="Missing required field"):
            validate_sample(d)

    def test_metadata_not_dict_raises(self):
        d = _valid_dict(metadata="not_a_dict")
        with pytest.raises(ValueError, match="metadata"):
            validate_sample(d)

    def test_non_string_label_raises(self):
        d = _valid_dict(label=1)
        with pytest.raises(ValueError):
            validate_sample(d)


# ---------------------------------------------------------------------------
# validate_jsonl
# ---------------------------------------------------------------------------

class TestValidateJsonl:
    def test_all_valid_returns_correct_count(self, tmp_path):
        records = [_valid_dict() for _ in range(5)]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, records)
        count, errors = validate_jsonl(p)
        assert count == 5
        assert errors == []

    def test_mixed_valid_invalid(self, tmp_path):
        records = [
            _valid_dict(),
            _valid_dict(label="INVALID"),
            _valid_dict(),
        ]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, records)
        count, errors = validate_jsonl(p)
        assert count == 2
        assert len(errors) == 1
        assert "Line 2" in errors[0]

    def test_empty_file_returns_zero_count(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        count, errors = validate_jsonl(p)
        assert count == 0
        assert errors == []


# ---------------------------------------------------------------------------
# has_description
# ---------------------------------------------------------------------------

class TestHasDescription:
    def test_non_empty_description_returns_true(self):
        s = BenchmarkSample(
            id="1", advertised_description="Something",
            prompt_body="body", label="benign", attack_type="none",
            source="deepset",
        )
        assert has_description(s) is True

    def test_empty_description_returns_false(self):
        s = BenchmarkSample(
            id="1", advertised_description="",
            prompt_body="body", label="benign", attack_type="none",
            source="deepset",
        )
        assert has_description(s) is False

    def test_whitespace_only_returns_false(self):
        s = BenchmarkSample(
            id="1", advertised_description="   ",
            prompt_body="body", label="benign", attack_type="none",
            source="deepset",
        )
        assert has_description(s) is False

    def test_dict_with_description(self):
        d = {"advertised_description": "Hello"}
        assert has_description(d) is True

    def test_dict_empty_description(self):
        d = {"advertised_description": ""}
        assert has_description(d) is False

    def test_dict_missing_key(self):
        d = {}
        assert has_description(d) is False


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        s = BenchmarkSample(
            id="abc-123",
            advertised_description="Describe the topic.",
            prompt_body="Tell me about AI.",
            label="benign",
            attack_type="none",
            source="spml",
            metadata={"tokens": 5, "language": "en", "severity": 2},
        )
        d = s.to_dict()
        s2 = BenchmarkSample.from_dict(d)
        assert s2.id == s.id
        assert s2.advertised_description == s.advertised_description
        assert s2.prompt_body == s.prompt_body
        assert s2.label == s.label
        assert s2.attack_type == s.attack_type
        assert s2.source == s.source
        assert s2.metadata == s.metadata

    def test_to_dict_contains_all_fields(self):
        s = BenchmarkSample(
            id="x", advertised_description="d", prompt_body="b",
            label="malicious", attack_type="jailbreak", source="adversarial",
        )
        d = s.to_dict()
        for key in ("id", "advertised_description", "prompt_body", "label", "attack_type", "source", "metadata"):
            assert key in d
