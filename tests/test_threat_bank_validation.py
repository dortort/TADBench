import json
from dataclasses import replace

import pytest

from tadbench.data.validation import load_threat_bank


def test_bank_must_be_explicit_nonempty_and_malicious(tmp_path, sample_data):
    with pytest.raises(ValueError, match="required"):
        load_threat_bank(None, sample_data)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("")
    with pytest.raises(ValueError, match="malicious-only"):
        load_threat_bank(bank, sample_data)
    bank.write_text(json.dumps(sample_data[0].to_dict()) + "\n")
    with pytest.raises(ValueError, match="malicious-only"):
        load_threat_bank(bank, [])


def test_bank_rejects_copied_content_with_new_ids(tmp_path, sample_data):
    threat = sample_data[-1]
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps(threat.to_dict()) + "\n")
    copied = replace(threat, id="new-id")
    with pytest.raises(ValueError, match="overlaps"):
        load_threat_bank(bank, [copied])
    assert load_threat_bank(bank, [replace(threat, id="other", prompt_body="unseen")]) == [threat]
