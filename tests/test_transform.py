"""Tests for tadbench.data.transform."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tadbench.data.schema import validate_sample
from tadbench.data.transform import (
    _label_from_int,
    _token_count,
    transform_deepset,
    transform_promptshield,
    transform_spml,
)
from tadbench.utils.io import read_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet_mock(tmp_path: Path, dataset_name: str, split: str, table: dict) -> None:
    """Write a fake Parquet directory that _load_parquet_split can read via pandas mock."""
    dataset_dir = tmp_path / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # We'll mock _load_parquet_split directly instead of writing real Parquet


def _mock_parquet_loader(table: dict):
    """Return a context-manager-compatible mock for _load_parquet_split."""
    def _loader(dataset_dir, split):
        return table
    return _loader


# ---------------------------------------------------------------------------
# _label_from_int
# ---------------------------------------------------------------------------

class TestLabelFromInt:
    def test_zero_is_benign(self):
        assert _label_from_int(0) == "benign"

    def test_one_is_malicious(self):
        assert _label_from_int(1) == "malicious"

    def test_string_zero(self):
        assert _label_from_int("0") == "benign"

    def test_string_one(self):
        assert _label_from_int("1") == "malicious"


# ---------------------------------------------------------------------------
# _token_count
# ---------------------------------------------------------------------------

class TestTokenCount:
    def test_counts_whitespace_tokens(self):
        assert _token_count("hello world foo") == 3

    def test_empty_string(self):
        assert _token_count("") == 0

    def test_single_word(self):
        assert _token_count("hello") == 1


# ---------------------------------------------------------------------------
# transform_promptshield
# ---------------------------------------------------------------------------

class TestTransformPromptshield:
    def _make_table(self, n: int = 6):
        prompts = [f"Prompt text {i}" for i in range(n)]
        labels = [i % 2 for i in range(n)]  # alternating 0/1
        return {"prompt": prompts, "label": labels}

    def test_output_file_created(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "promptshield").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = self._make_table(4)
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_promptshield(raw_dir, out_dir)

        # At least one output file should exist
        jsonl_files = list(out_dir.glob("promptshield_*.jsonl"))
        assert len(jsonl_files) >= 1

    def test_schema_conformance(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "promptshield").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = self._make_table(6)
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_promptshield(raw_dir, out_dir)

        for jf in out_dir.glob("promptshield_*.jsonl"):
            records = read_jsonl(jf)
            for r in records:
                sample = validate_sample(r)
                assert sample.source == "promptshield"
                assert sample.advertised_description == ""
                assert sample.label in ("benign", "malicious")

    def test_label_mapping_correct(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "promptshield").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = {"prompt": ["benign text", "malicious text"], "label": [0, 1]}
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_promptshield(raw_dir, out_dir)

        for jf in out_dir.glob("promptshield_*.jsonl"):
            records = read_jsonl(jf)
            labels = {r["prompt_body"]: r["label"] for r in records}
            assert labels["benign text"] == "benign"
            assert labels["malicious text"] == "malicious"

    def test_skips_missing_split(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "promptshield").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        # Return None to simulate missing file
        with patch("tadbench.data.transform._load_parquet_split", return_value=None):
            transform_promptshield(raw_dir, out_dir)

        assert list(out_dir.glob("*.jsonl")) == []


# ---------------------------------------------------------------------------
# transform_deepset
# ---------------------------------------------------------------------------

class TestTransformDeepset:
    def test_schema_conformance(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "deepset").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = {
            "text": ["Good prompt", "Bad prompt ignore instructions"],
            "label": [0, 1],
        }
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_deepset(raw_dir, out_dir)

        for jf in out_dir.glob("deepset_*.jsonl"):
            records = read_jsonl(jf)
            for r in records:
                s = validate_sample(r)
                assert s.source == "deepset"
                assert s.advertised_description == ""

    def test_token_count_in_metadata(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "deepset").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = {"text": ["one two three four five"], "label": [0]}
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_deepset(raw_dir, out_dir)

        for jf in out_dir.glob("deepset_*.jsonl"):
            records = read_jsonl(jf)
            assert records[0]["metadata"]["tokens"] == 5


# ---------------------------------------------------------------------------
# transform_spml
# ---------------------------------------------------------------------------

class TestTransformSpml:
    def test_system_prompt_becomes_description(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "spml").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = {
            "User Prompt": ["Do this task"],
            "System Prompt": ["You are a helpful assistant"],
            "Prompt injection": [0],
            "Degree": [None],
        }
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_spml(raw_dir, out_dir)

        for jf in out_dir.glob("spml_*.jsonl"):
            records = read_jsonl(jf)
            assert records[0]["advertised_description"] == "You are a helpful assistant"
            assert records[0]["source"] == "spml"

    def test_degree_stored_as_severity(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "spml").mkdir(parents=True)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        table = {
            "User Prompt": ["Inject here"],
            "System Prompt": ["Assistant"],
            "Prompt injection": [1],
            "Degree": [3],
        }
        with patch("tadbench.data.transform._load_parquet_split", return_value=table):
            transform_spml(raw_dir, out_dir)

        for jf in out_dir.glob("spml_*.jsonl"):
            records = read_jsonl(jf)
            assert records[0]["metadata"].get("severity") == 3
