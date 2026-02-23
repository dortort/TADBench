"""Download public datasets from HuggingFace and save as Parquet."""

from __future__ import annotations

import sys
from pathlib import Path

from tadbench.config import HF_DATASETS, RAW_DIR


def download_dataset(name: str, output_dir: Path) -> None:
    """Download a single HuggingFace dataset and save all splits as Parquet.

    Args:
        name: Key in HF_DATASETS (e.g. "promptshield").
        output_dir: Root directory under which ``{name}/`` is created.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    cfg = HF_DATASETS[name]
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Downloading '{cfg['path']}' (revision={cfg['revision']}) ...")
    ds = load_dataset(cfg["path"], revision=cfg["revision"])

    for split_name, split_data in ds.items():
        out_path = dataset_dir / f"{split_name}.parquet"
        split_data.to_parquet(str(out_path))
        print(f"[download]   {split_name}: {len(split_data):,} rows -> {out_path}")

    print(f"[download] Done: {name}")


def download_all(output_dir: Path | str | None = None) -> None:
    """Download all three benchmark datasets to ``output_dir``.

    Args:
        output_dir: Directory in which per-dataset subdirectories are created.
                    Defaults to ``tadbench.config.RAW_DIR``.
    """
    out = Path(output_dir) if output_dir is not None else RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    for name in HF_DATASETS:
        download_dataset(name, out)

    print(f"\n[download] All datasets saved to: {out}")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    download_all(dest)
