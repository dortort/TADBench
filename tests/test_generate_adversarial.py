"""Tests for tadbench.data.generate_adversarial."""

from __future__ import annotations

import random

import pytest

from tadbench.data.generate_adversarial import (
    _generate_iccd_evasion,
    _generate_ipsod_evasion,
    _generate_tps_evasion,
    generate_adversarial,
)
from tadbench.data.schema import validate_sample
from tadbench.utils.io import read_jsonl


class TestGenerateTpsEvasion:
    def test_returns_correct_count(self):
        rng = random.Random(0)
        samples = _generate_tps_evasion(10, rng)
        assert len(samples) == 10

    def test_all_malicious(self):
        rng = random.Random(0)
        samples = _generate_tps_evasion(20, rng)
        assert all(s["label"] == "malicious" for s in samples)

    def test_correct_attack_type(self):
        rng = random.Random(0)
        samples = _generate_tps_evasion(10, rng)
        assert all(s["attack_type"] == "tps_evasion" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(0)
        samples = _generate_tps_evasion(10, rng)
        assert all(s["advertised_description"].strip() for s in samples)

    def test_schema_valid(self):
        rng = random.Random(0)
        samples = _generate_tps_evasion(5, rng)
        for s in samples:
            v = validate_sample(s)
            assert v.source == "adversarial"


class TestGenerateIpsodEvasion:
    def test_returns_correct_count(self):
        rng = random.Random(1)
        samples = _generate_ipsod_evasion(10, rng)
        assert len(samples) == 10

    def test_all_malicious(self):
        rng = random.Random(1)
        samples = _generate_ipsod_evasion(20, rng)
        assert all(s["label"] == "malicious" for s in samples)

    def test_correct_attack_type(self):
        rng = random.Random(1)
        samples = _generate_ipsod_evasion(10, rng)
        assert all(s["attack_type"] == "ipsod_evasion" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(1)
        samples = _generate_ipsod_evasion(10, rng)
        assert all(s["advertised_description"].strip() for s in samples)

    def test_no_empty_prompt_body(self):
        rng = random.Random(1)
        samples = _generate_ipsod_evasion(10, rng)
        assert all(s["prompt_body"].strip() for s in samples)


class TestGenerateIccdEvasion:
    def test_returns_correct_count(self):
        rng = random.Random(2)
        samples = _generate_iccd_evasion(10, rng)
        assert len(samples) == 10

    def test_all_malicious(self):
        rng = random.Random(2)
        samples = _generate_iccd_evasion(20, rng)
        assert all(s["label"] == "malicious" for s in samples)

    def test_correct_attack_type(self):
        rng = random.Random(2)
        samples = _generate_iccd_evasion(10, rng)
        assert all(s["attack_type"] == "iccd_evasion" for s in samples)

    def test_all_have_descriptions(self):
        rng = random.Random(2)
        samples = _generate_iccd_evasion(10, rng)
        assert all(s["advertised_description"].strip() for s in samples)


class TestGenerateAdversarial:
    def test_output_files_created(self, tmp_path):
        result = generate_adversarial(output_dir=tmp_path, n_tps=5, n_ipsod=5, n_iccd=5)
        assert (tmp_path / "adversarial_tps_evasion.jsonl").exists()
        assert (tmp_path / "adversarial_ipsod_evasion.jsonl").exists()
        assert (tmp_path / "adversarial_iccd_evasion.jsonl").exists()

    def test_returns_dict_with_three_keys(self, tmp_path):
        result = generate_adversarial(output_dir=tmp_path, n_tps=5, n_ipsod=5, n_iccd=5)
        assert set(result.keys()) == {"tps_evasion", "ipsod_evasion", "iccd_evasion"}

    def test_counts_match_requested(self, tmp_path):
        result = generate_adversarial(output_dir=tmp_path, n_tps=7, n_ipsod=6, n_iccd=5)
        assert len(result["tps_evasion"]) == 7
        assert len(result["ipsod_evasion"]) == 6
        assert len(result["iccd_evasion"]) == 5

    def test_all_labeled_malicious(self, tmp_path):
        result = generate_adversarial(output_dir=tmp_path, n_tps=5, n_ipsod=5, n_iccd=5)
        all_samples = (
            result["tps_evasion"] + result["ipsod_evasion"] + result["iccd_evasion"]
        )
        assert all(s["label"] == "malicious" for s in all_samples)

    def test_all_have_descriptions(self, tmp_path):
        result = generate_adversarial(output_dir=tmp_path, n_tps=5, n_ipsod=5, n_iccd=5)
        all_samples = (
            result["tps_evasion"] + result["ipsod_evasion"] + result["iccd_evasion"]
        )
        assert all(s["advertised_description"].strip() for s in all_samples)

    def test_jsonl_files_valid(self, tmp_path):
        generate_adversarial(output_dir=tmp_path, n_tps=5, n_ipsod=5, n_iccd=5)
        for fname in (
            "adversarial_tps_evasion.jsonl",
            "adversarial_ipsod_evasion.jsonl",
            "adversarial_iccd_evasion.jsonl",
        ):
            records = read_jsonl(tmp_path / fname)
            assert len(records) == 5
            for r in records:
                validate_sample(r)

    def test_reproducible_with_seed(self, tmp_path):
        r1 = generate_adversarial(output_dir=tmp_path / "a", n_tps=5, n_ipsod=5, n_iccd=5, seed=7)
        r2 = generate_adversarial(output_dir=tmp_path / "b", n_tps=5, n_ipsod=5, n_iccd=5, seed=7)
        for key in ("tps_evasion", "ipsod_evasion", "iccd_evasion"):
            bodies1 = [s["prompt_body"] for s in r1[key]]
            bodies2 = [s["prompt_body"] for s in r2[key]]
            assert bodies1 == bodies2
