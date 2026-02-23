"""Tests for tadbench.data.download."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from tadbench.data.download import download_all, download_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_split(n_rows: int = 5) -> MagicMock:
    """Return a mock HuggingFace split that supports to_parquet() and len()."""
    mock_split = MagicMock()
    mock_split.__len__ = MagicMock(return_value=n_rows)
    mock_split.to_parquet = MagicMock()
    return mock_split


def _make_mock_dataset(*split_names: str) -> MagicMock:
    """Return a mock DatasetDict with the given split names."""
    mock_ds = {name: _make_mock_split() for name in split_names}
    return mock_ds


def _fake_datasets_module(load_dataset_mock: MagicMock) -> ModuleType:
    """Build a minimal fake 'datasets' module with a mocked load_dataset."""
    fake_mod = ModuleType("datasets")
    fake_mod.load_dataset = load_dataset_mock  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# download_dataset
# ---------------------------------------------------------------------------

class TestDownloadDataset:
    def test_creates_output_directory(self, tmp_path):
        mock_ds = _make_mock_dataset("train", "test")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_dataset("deepset", output_dir=tmp_path)
        assert (tmp_path / "deepset").is_dir()

    def test_calls_to_parquet_for_each_split(self, tmp_path):
        mock_ds = _make_mock_dataset("train", "test")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_dataset("deepset", output_dir=tmp_path)
        # to_parquet should have been called once per split
        for split_name, split_obj in mock_ds.items():
            split_obj.to_parquet.assert_called_once()

    def test_passes_correct_hf_path(self, tmp_path):
        mock_ds = _make_mock_dataset("train")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_dataset("promptshield", output_dir=tmp_path)
        call_args = mock_load.call_args
        assert (
            "hendzh/PromptShield" in call_args[0]
            or call_args[1].get("path") == "hendzh/PromptShield"
            or call_args[0][0] == "hendzh/PromptShield"
        )

    def test_unknown_dataset_raises_key_error(self, tmp_path):
        # Inject a fake datasets module so the lazy import succeeds; the
        # KeyError should come from HF_DATASETS["nonexistent_dataset"] lookup.
        fake_mod = _fake_datasets_module(MagicMock())
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            with pytest.raises(KeyError):
                download_dataset("nonexistent_dataset", output_dir=tmp_path)

    def test_datasets_import_error_raises_import_error(self, tmp_path):
        # Simulate datasets library not installed
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises((ImportError, TypeError)):
                download_dataset("deepset", output_dir=tmp_path)


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------

class TestDownloadAll:
    def test_creates_subdirectory_for_each_dataset(self, tmp_path):
        mock_ds = _make_mock_dataset("train", "test")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_all(output_dir=tmp_path)

        # Should have created one dir per dataset in HF_DATASETS
        from tadbench.config import HF_DATASETS
        for name in HF_DATASETS:
            assert (tmp_path / name).is_dir(), f"Expected directory for {name}"

    def test_creates_output_dir_if_missing(self, tmp_path):
        target = tmp_path / "new_raw"
        assert not target.exists()
        mock_ds = _make_mock_dataset("train")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_all(output_dir=target)
        assert target.exists()

    def test_calls_load_dataset_for_each_dataset(self, tmp_path):
        mock_ds = _make_mock_dataset("train")
        mock_load = MagicMock(return_value=mock_ds)
        fake_mod = _fake_datasets_module(mock_load)
        with patch.dict("sys.modules", {"datasets": fake_mod}):
            download_all(output_dir=tmp_path)

        from tadbench.config import HF_DATASETS
        assert mock_load.call_count == len(HF_DATASETS)
