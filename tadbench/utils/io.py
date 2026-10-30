"""JSONL read/write helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file."""
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(data: list[dict], path: str | Path) -> None:
    """Write records to a JSONL file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stream_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream records from a JSONL file one at a time (memory-efficient)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def json_safe(value):
    """Use null for undefined metrics and strings for infinite thresholds."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    return value
