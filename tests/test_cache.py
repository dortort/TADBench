"""Tests for tadbench.embedders.cache.EmbeddingCache."""

from __future__ import annotations

import numpy as np
import pytest

from tadbench.embedders.cache import EmbeddingCache


class TestEmbeddingCacheHitMiss:
    def test_first_call_is_miss(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["hello"])
        # After 1 call with 1 text: 0 hits, 1 total => hit_rate = 0
        assert cache.hit_rate == 0.0

    def test_second_call_same_text_is_hit(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["hello"])
        cache.encode(["hello"])
        # 2 total, 1 hit => 0.5
        assert cache.hit_rate == 0.5

    def test_all_cached_returns_one(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        texts = ["a", "b", "c"]
        cache.encode(texts)     # 3 misses
        cache.encode(texts)     # 3 hits
        # 6 total, 3 hits
        assert cache.hit_rate == 0.5

    def test_different_texts_no_hit(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["text one"])
        cache.encode(["text two"])
        # 2 total, 0 hits
        assert cache.hit_rate == 0.0

    def test_partial_hit(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["cached"])
        cache.encode(["cached", "new"])
        # Total: 1 + 2 = 3; hits: 0 + 1 = 1; rate = 1/3
        assert abs(cache.hit_rate - 1 / 3) < 1e-9


class TestEmbeddingCacheValues:
    def test_cached_value_matches_direct_encode(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        text = "consistent text"
        direct = mock_embedder.encode([text])[0]
        cached_first = cache.encode([text])[0]
        cached_second = cache.encode([text])[0]
        np.testing.assert_array_equal(direct, cached_first)
        np.testing.assert_array_equal(direct, cached_second)

    def test_encode_returns_correct_shape(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        out = cache.encode(["a", "b", "c"])
        assert out.shape == (3, mock_embedder.dimension)

    def test_encode_single_returns_1d(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        out = cache.encode_single("hello")
        assert out.ndim == 1
        assert out.shape == (mock_embedder.dimension,)


class TestEmbeddingCacheClear:
    def test_clear_resets_hit_rate(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["hello"])
        cache.encode(["hello"])   # creates a hit
        cache.clear()
        assert cache.hit_rate == 0.0

    def test_clear_removes_in_memory_store(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        cache.encode(["hello"])
        cache.clear()
        # After clear, re-encoding should be a miss
        cache.encode(["hello"])
        assert cache.hit_rate == 0.0

    def test_clear_deletes_disk_file(self, mock_embedder, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = EmbeddingCache(mock_embedder, cache_dir=str(cache_dir))
        cache.encode(["hello"])
        # Force save to disk
        cache._save()
        disk_files_before = list(cache_dir.glob("*.npz"))
        assert len(disk_files_before) >= 1
        cache.clear()
        disk_files_after = list(cache_dir.glob("*.npz"))
        assert len(disk_files_after) == 0


class TestEmbeddingCacheDelegation:
    def test_dimension_delegates_to_embedder(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        assert cache.dimension == mock_embedder.dimension

    def test_model_name_delegates_to_embedder(self, mock_embedder, tmp_path):
        cache = EmbeddingCache(mock_embedder, cache_dir=str(tmp_path / "cache"))
        assert cache.model_name == mock_embedder.model_name


class TestEmbeddingCachePersistence:
    def test_encode_save_clear_reload(self, mock_embedder, tmp_path):
        """Encode some texts, force save, clear memory, reload from disk."""
        cache_dir = tmp_path / "cache"
        cache = EmbeddingCache(mock_embedder, cache_dir=str(cache_dir))

        texts = ["persist me", "also persist"]
        expected = mock_embedder.encode(texts)

        cache.encode(texts)
        cache._save()  # force persistence

        # Clear in-memory store (simulate new process by loading fresh)
        cache2 = EmbeddingCache(mock_embedder, cache_dir=str(cache_dir))
        result = cache2.encode(texts)

        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        # All should be hits since loaded from disk
        assert cache2.hit_rate == 1.0
