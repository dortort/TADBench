"""Transform raw HuggingFace datasets into unified BenchmarkSample JSONL."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from tadbench.config import PROCESSED_DIR, RAW_DIR
from tadbench.utils.io import write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


def _make_record(
    prompt_body: str,
    label: str,
    attack_type: str,
    source: str,
    advertised_description: str = "",
    severity: int | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    tokens = _token_count(prompt_body)
    metadata: dict[str, Any] = {"tokens": tokens, "language": "en"}
    if severity is not None:
        metadata["severity"] = severity
    return {
        "id": sample_id or str(uuid.uuid4()),
        "advertised_description": advertised_description,
        "prompt_body": prompt_body,
        "label": label,
        "attack_type": attack_type,
        "source": source,
        "metadata": metadata,
    }


def _label_from_int(val: int) -> str:
    """Map integer 0/1 label to 'benign'/'malicious'."""
    return "malicious" if int(val) == 1 else "benign"


def _attack_from_label(label: str) -> str:
    return "direct_injection" if label == "malicious" else "none"


# ---------------------------------------------------------------------------
# Per-dataset transformers
# ---------------------------------------------------------------------------

def _load_parquet_split(dataset_dir: Path, split: str):
    """Load a single split Parquet file using the datasets library."""
    parquet_path = dataset_dir / f"{split}.parquet"
    if not parquet_path.exists():
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import]
        return pq.read_table(str(parquet_path)).to_pydict()
    except ImportError:
        try:
            import pandas as pd  # type: ignore[import]
            df = pd.read_parquet(str(parquet_path))
            return df.to_dict(orient="list")
        except ImportError as exc:
            raise ImportError(
                "Either 'pyarrow' or 'pandas' is required to read Parquet files."
            ) from exc


def _iter_rows(table: dict[str, list], *fields: str):
    """Yield tuples of field values across all rows."""
    n = len(next(iter(table.values())))
    for i in range(n):
        yield tuple(table[f][i] for f in fields)


def transform_promptshield(raw_dir: Path, output_dir: Path) -> None:
    """Transform PromptShield splits -> unified JSONL.

    Field mapping:
      - prompt        -> prompt_body
      - label (0/1)   -> benign/malicious
      - advertised_description = "" (not available)
      - attack_type   = "none" | "direct_injection"
    """
    dataset_dir = raw_dir / "promptshield"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "validation", "test"]
    # Some HF datasets use "validation" as the key; handle both
    split_file_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split, parquet_split in split_file_map.items():
        table = _load_parquet_split(dataset_dir, parquet_split)
        if table is None:
            # Try alternate name
            alt = "val" if parquet_split == "validation" else parquet_split
            table = _load_parquet_split(dataset_dir, alt)
        if table is None:
            print(f"[transform] promptshield/{parquet_split}: not found, skipping")
            continue

        # Determine available columns (handle case variations)
        cols = set(table.keys())
        prompt_col = next((c for c in ("prompt", "Prompt", "text", "Text") if c in cols), None)
        label_col = next((c for c in ("label", "Label") if c in cols), None)

        if prompt_col is None or label_col is None:
            print(f"[transform] promptshield/{split}: unexpected columns {cols}, skipping")
            continue

        records = []
        for prompt, label_val in _iter_rows(table, prompt_col, label_col):
            prompt = str(prompt) if prompt is not None else ""
            lbl = _label_from_int(label_val)
            records.append(_make_record(
                prompt_body=prompt,
                label=lbl,
                attack_type=_attack_from_label(lbl),
                source="promptshield",
                advertised_description="",
            ))

        out_split = "val" if split == "validation" else split
        out_path = output_dir / f"promptshield_{out_split}.jsonl"
        write_jsonl(records, out_path)
        malicious = sum(1 for r in records if r["label"] == "malicious")
        avg_tok = sum(r["metadata"]["tokens"] for r in records) / max(len(records), 1)
        print(
            f"[transform] promptshield/{split}: {len(records):,} samples "
            f"({malicious:,} malicious, avg {avg_tok:.0f} tokens) -> {out_path}"
        )


def transform_deepset(raw_dir: Path, output_dir: Path) -> None:
    """Transform deepset/prompt-injections -> unified JSONL.

    Field mapping:
      - text          -> prompt_body
      - label (0/1)   -> benign/malicious
      - advertised_description = ""
    """
    dataset_dir = raw_dir / "deepset"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        table = _load_parquet_split(dataset_dir, split)
        if table is None:
            print(f"[transform] deepset/{split}: not found, skipping")
            continue

        cols = set(table.keys())
        text_col = next((c for c in ("text", "Text", "prompt", "Prompt") if c in cols), None)
        label_col = next((c for c in ("label", "Label") if c in cols), None)

        if text_col is None or label_col is None:
            print(f"[transform] deepset/{split}: unexpected columns {cols}, skipping")
            continue

        records = []
        for text, label_val in _iter_rows(table, text_col, label_col):
            text = str(text) if text is not None else ""
            lbl = _label_from_int(label_val)
            records.append(_make_record(
                prompt_body=text,
                label=lbl,
                attack_type=_attack_from_label(lbl),
                source="deepset",
                advertised_description="",
            ))

        out_path = output_dir / f"deepset_{split}.jsonl"
        write_jsonl(records, out_path)
        malicious = sum(1 for r in records if r["label"] == "malicious")
        avg_tok = sum(r["metadata"]["tokens"] for r in records) / max(len(records), 1)
        print(
            f"[transform] deepset/{split}: {len(records):,} samples "
            f"({malicious:,} malicious, avg {avg_tok:.0f} tokens) -> {out_path}"
        )


def transform_spml(raw_dir: Path, output_dir: Path) -> None:
    """Transform SPML_Chatbot_Prompt_Injection -> unified JSONL.

    Field mapping:
      - User Prompt     -> prompt_body
      - System Prompt   -> advertised_description
      - Prompt injection (0/1) -> label
      - Degree          -> metadata.severity
    """
    dataset_dir = raw_dir / "spml"
    output_dir.mkdir(parents=True, exist_ok=True)

    # SPML has a single "train" split; splits.py creates train/val/test
    for split in ("train", "test"):
        table = _load_parquet_split(dataset_dir, split)
        if table is None:
            print(f"[transform] spml/{split}: not found, skipping")
            continue

        cols = set(table.keys())

        # Column name detection (handle possible spacing/casing variants)
        user_col = next(
            (c for c in ("User Prompt", "user_prompt", "user prompt", "prompt", "Prompt") if c in cols),
            None,
        )
        system_col = next(
            (c for c in ("System Prompt", "system_prompt", "system prompt") if c in cols),
            None,
        )
        label_col = next(
            (c for c in ("Prompt injection", "prompt_injection", "label", "Label") if c in cols),
            None,
        )
        degree_col = next(
            (c for c in ("Degree", "degree", "severity", "Severity") if c in cols),
            None,
        )

        if user_col is None or label_col is None:
            print(f"[transform] spml/{split}: unexpected columns {cols}, skipping")
            continue

        iter_fields = [user_col, label_col]
        if system_col:
            iter_fields.append(system_col)
        if degree_col:
            iter_fields.append(degree_col)

        records = []
        for row in zip(*[table[f] for f in iter_fields]):
            user_prompt = str(row[0]) if row[0] is not None else ""
            label_val = row[1]
            sys_prompt = str(row[2]) if system_col and row[2] is not None else ""
            degree_val = row[3] if degree_col and len(row) > 3 else None

            lbl = _label_from_int(label_val)
            severity: int | None = None
            if degree_val is not None:
                try:
                    severity = int(degree_val)
                except (TypeError, ValueError):
                    severity = None

            records.append(_make_record(
                prompt_body=user_prompt,
                label=lbl,
                attack_type=_attack_from_label(lbl),
                source="spml",
                advertised_description=sys_prompt,
                severity=severity,
            ))

        out_path = output_dir / f"spml_{split}.jsonl"
        write_jsonl(records, out_path)
        malicious = sum(1 for r in records if r["label"] == "malicious")
        avg_tok = sum(r["metadata"]["tokens"] for r in records) / max(len(records), 1)
        print(
            f"[transform] spml/{split}: {len(records):,} samples "
            f"({malicious:,} malicious, avg {avg_tok:.0f} tokens) -> {out_path}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_all(
    raw_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Transform all raw datasets to unified JSONL.

    Args:
        raw_dir: Directory containing per-dataset Parquet subdirs.
                 Defaults to ``tadbench.config.RAW_DIR``.
        output_dir: Directory to write JSONL files.
                    Defaults to ``tadbench.config.PROCESSED_DIR``.
    """
    raw = Path(raw_dir) if raw_dir is not None else RAW_DIR
    out = Path(output_dir) if output_dir is not None else PROCESSED_DIR

    print("[transform] PromptShield ...")
    transform_promptshield(raw, out)

    print("[transform] deepset ...")
    transform_deepset(raw, out)

    print("[transform] SPML ...")
    transform_spml(raw, out)

    print(f"\n[transform] All datasets written to: {out}")


if __name__ == "__main__":
    raw_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    transform_all(raw_arg, out_arg)
