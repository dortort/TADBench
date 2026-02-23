"""Unified sample schema and validation for TADBench."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tadbench.config import VALID_ATTACK_TYPES, VALID_LABELS, VALID_SOURCES
from tadbench.utils.io import stream_jsonl


@dataclass
class BenchmarkSample:
    """A single benchmark sample in the unified TADBench schema."""

    id: str
    advertised_description: str  # Empty string when unavailable; required for Technique 3/cascade
    prompt_body: str
    label: str           # "benign" | "malicious"
    attack_type: str     # See VALID_ATTACK_TYPES in config.py
    source: str          # See VALID_SOURCES in config.py
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: tokens (int), language (str), severity (int | None)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary suitable for JSONL output."""
        return {
            "id": self.id,
            "advertised_description": self.advertised_description,
            "prompt_body": self.prompt_body,
            "label": self.label,
            "attack_type": self.attack_type,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkSample":
        """Deserialize from a dictionary (e.g., parsed from JSONL)."""
        return cls(
            id=d["id"],
            advertised_description=d["advertised_description"],
            prompt_body=d["prompt_body"],
            label=d["label"],
            attack_type=d["attack_type"],
            source=d["source"],
            metadata=d.get("metadata", {}),
        )


def _check_required_str(d: dict, key: str, errors: list[str]) -> str:
    """Validate that a key exists and is a non-None string."""
    val = d.get(key)
    if val is None:
        errors.append(f"Missing required field: '{key}'")
        return ""
    if not isinstance(val, str):
        errors.append(f"Field '{key}' must be a string, got {type(val).__name__}")
        return ""
    return val


def validate_sample(sample: dict) -> BenchmarkSample:
    """Validate a raw dictionary and return a BenchmarkSample.

    Raises ValueError with a descriptive message if validation fails.
    """
    errors: list[str] = []

    sample_id = _check_required_str(sample, "id", errors)
    advertised_description = _check_required_str(sample, "advertised_description", errors)
    prompt_body = _check_required_str(sample, "prompt_body", errors)
    label = _check_required_str(sample, "label", errors)
    attack_type = _check_required_str(sample, "attack_type", errors)
    source = _check_required_str(sample, "source", errors)

    # Allowed-value checks (only when field is present)
    if label and label not in VALID_LABELS:
        errors.append(f"Invalid label '{label}'; must be one of {sorted(VALID_LABELS)}")
    if attack_type and attack_type not in VALID_ATTACK_TYPES:
        errors.append(
            f"Invalid attack_type '{attack_type}'; must be one of {sorted(VALID_ATTACK_TYPES)}"
        )
    if source and source not in VALID_SOURCES:
        errors.append(f"Invalid source '{source}'; must be one of {sorted(VALID_SOURCES)}")

    # metadata must be a dict (can be empty)
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"Field 'metadata' must be a dict, got {type(metadata).__name__}")
        metadata = {}

    if errors:
        raise ValueError(
            f"Sample validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # Auto-generate id if empty string was provided
    if not sample_id:
        sample_id = str(uuid.uuid4())

    return BenchmarkSample(
        id=sample_id,
        advertised_description=advertised_description,
        prompt_body=prompt_body,
        label=label,
        attack_type=attack_type,
        source=source,
        metadata=metadata,
    )


def validate_jsonl(path: str | Path) -> tuple[int, list[str]]:
    """Validate every record in a JSONL file.

    Returns (valid_count, error_messages).  A file with 0 errors has
    valid_count == total records and an empty error list.
    """
    path = Path(path)
    valid_count = 0
    error_messages: list[str] = []

    for line_num, record in enumerate(stream_jsonl(path), start=1):
        try:
            validate_sample(record)
            valid_count += 1
        except ValueError as exc:
            error_messages.append(f"Line {line_num}: {exc}")

    return valid_count, error_messages


def has_description(sample: BenchmarkSample | dict) -> bool:
    """Return True if the sample has a non-empty advertised_description.

    Accepts both BenchmarkSample instances and plain dicts.
    """
    if isinstance(sample, dict):
        return bool(sample.get("advertised_description", "").strip())
    return bool(sample.advertised_description.strip())
