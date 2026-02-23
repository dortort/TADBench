"""Tests for tadbench.data.generate_synthetic."""

from __future__ import annotations

from pathlib import Path

import pytest

from tadbench.data.generate_synthetic import (
    _generate_benign_pairs,
    _generate_hard_negatives,
    _generate_malicious_mismatch,
    generate_synthetic,
)
from tadbench.data.schema import validate_sample
from tadbench.utils.io import read_jsonl

import random


# ---------------------------------------------------------------------------
# Private generators — small counts for speed
# ---------------------------------------------------------------------------

class TestGenerateBenignPairs:
    def test_returns_correct_count(self):
        rng = random.Random(0)
        samples = _generate_benign_pairs(10, rng)
        assert len(samples) == 10

    def test_all_benign(self):
        rng = random.Random(0)
        samples = _generate_benign_pairs(20, rng)
        assert all(s["label"] == "benign" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(0)
        samples = _generate_benign_pairs(20, rng)
        assert all(s["advertised_description"].strip() for s in samples)

    def test_no_empty_prompt_body(self):
        rng = random.Random(0)
        samples = _generate_benign_pairs(20, rng)
        assert all(s["prompt_body"].strip() for s in samples)

    def test_schema_valid(self):
        rng = random.Random(0)
        samples = _generate_benign_pairs(5, rng)
        for s in samples:
            validated = validate_sample(s)
            assert validated.label == "benign"


class TestGenerateMaliciousMismatch:
    def test_returns_correct_count(self):
        rng = random.Random(1)
        samples = _generate_malicious_mismatch(10, rng)
        assert len(samples) == 10

    def test_all_malicious(self):
        rng = random.Random(1)
        samples = _generate_malicious_mismatch(20, rng)
        assert all(s["label"] == "malicious" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(1)
        samples = _generate_malicious_mismatch(20, rng)
        assert all(s["advertised_description"].strip() for s in samples)

    def test_no_empty_prompt_body(self):
        rng = random.Random(1)
        samples = _generate_malicious_mismatch(20, rng)
        assert all(s["prompt_body"].strip() for s in samples)


class TestGenerateHardNegatives:
    def test_returns_correct_count(self):
        rng = random.Random(2)
        samples = _generate_hard_negatives(10, rng)
        assert len(samples) == 10

    def test_all_benign(self):
        rng = random.Random(2)
        samples = _generate_hard_negatives(10, rng)
        assert all(s["label"] == "benign" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(2)
        samples = _generate_hard_negatives(10, rng)
        assert all(s["advertised_description"].strip() for s in samples)


# ---------------------------------------------------------------------------
# generate_synthetic — small counts
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_output_file_created(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        generate_synthetic(output_path=out, n_benign=5, n_malicious=5, n_hard_negatives=5)
        assert out.exists()

    def test_total_count_correct(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        samples = generate_synthetic(
            output_path=out, n_benign=10, n_malicious=8, n_hard_negatives=6
        )
        assert len(samples) == 24

    def test_label_distribution(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        samples = generate_synthetic(
            output_path=out, n_benign=10, n_malicious=8, n_hard_negatives=6
        )
        benign = sum(1 for s in samples if s["label"] == "benign")
        malicious = sum(1 for s in samples if s["label"] == "malicious")
        # benign = n_benign + n_hard_negatives = 16, malicious = 8
        assert benign == 16
        assert malicious == 8

    def test_all_have_descriptions(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        samples = generate_synthetic(
            output_path=out, n_benign=5, n_malicious=5, n_hard_negatives=5
        )
        assert all(s["advertised_description"].strip() for s in samples)

    def test_no_empty_prompt_body(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        samples = generate_synthetic(
            output_path=out, n_benign=5, n_malicious=5, n_hard_negatives=5
        )
        assert all(s["prompt_body"].strip() for s in samples)

    def test_schema_valid(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        samples = generate_synthetic(
            output_path=out, n_benign=5, n_malicious=5, n_hard_negatives=5
        )
        for s in samples:
            validated = validate_sample(s)
            assert validated.source == "synthetic_mismatch"

    def test_output_is_valid_jsonl(self, tmp_path):
        out = tmp_path / "synthetic.jsonl"
        generate_synthetic(
            output_path=out, n_benign=5, n_malicious=5, n_hard_negatives=5
        )
        records = read_jsonl(out)
        assert len(records) == 15

    def test_reproducible_with_same_seed(self, tmp_path):
        out1 = tmp_path / "a.jsonl"
        out2 = tmp_path / "b.jsonl"
        s1 = generate_synthetic(output_path=out1, n_benign=5, n_malicious=5, n_hard_negatives=5, seed=99)
        s2 = generate_synthetic(output_path=out2, n_benign=5, n_malicious=5, n_hard_negatives=5, seed=99)
        assert [x["prompt_body"] for x in s1] == [x["prompt_body"] for x in s2]
