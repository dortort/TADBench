"""Tests for tadbench.embedders (protocol compliance and SentenceTransformerEmbedder)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tadbench.embedders.protocol import EmbedderProtocol


# ---------------------------------------------------------------------------
# MockEmbedder protocol compliance (from conftest)
# ---------------------------------------------------------------------------

class TestMockEmbedderProtocol:
    def test_implements_protocol(self, mock_embedder):
        assert isinstance(mock_embedder, EmbedderProtocol)

    def test_encode_returns_2d_array(self, mock_embedder):
        texts = ["hello", "world", "foo"]
        out = mock_embedder.encode(texts)
        assert out.ndim == 2
        assert out.shape == (3, 32)

    def test_encode_single_returns_1d_array(self, mock_embedder):
        out = mock_embedder.encode_single("hello world")
        assert out.ndim == 1
        assert out.shape == (32,)

    def test_dimension_property(self, mock_embedder):
        assert mock_embedder.dimension == 32

    def test_model_name_property(self, mock_embedder):
        assert isinstance(mock_embedder.model_name, str)
        assert len(mock_embedder.model_name) > 0

    def test_deterministic_for_same_text(self, mock_embedder):
        v1 = mock_embedder.encode_single("consistent text")
        v2 = mock_embedder.encode_single("consistent text")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self, mock_embedder):
        v1 = mock_embedder.encode_single("text one")
        v2 = mock_embedder.encode_single("text two")
        assert not np.allclose(v1, v2)

    def test_encode_float32_dtype(self, mock_embedder):
        out = mock_embedder.encode(["test"])
        assert out.dtype == np.float32

    def test_encode_unit_vectors(self, mock_embedder):
        out = mock_embedder.encode(["normalize me", "also this"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_empty_list(self, mock_embedder):
        out = mock_embedder.encode([])
        # Should return a 2D array with 0 rows
        assert out.ndim == 2
        assert out.shape[0] == 0

    def test_encode_single_matches_encode_batch(self, mock_embedder):
        text = "batch vs single"
        single = mock_embedder.encode_single(text)
        batch = mock_embedder.encode([text])[0]
        np.testing.assert_array_equal(single, batch)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — tier resolution WITHOUT loading real model
# ---------------------------------------------------------------------------

class TestSentenceTransformerEmbedderMocked:
    """Test SentenceTransformerEmbedder without actually loading a model."""

    def _make_mock_model(self, dim: int = 384) -> MagicMock:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = dim
        # encode() returns a numpy array of the right shape
        def fake_encode(texts, **kwargs):
            n = len(texts)
            vecs = np.random.RandomState(0).randn(n, dim).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
        mock_model.encode = fake_encode
        return mock_model

    def test_tier_small_resolves_correct_model_name(self):
        mock_model = self._make_mock_model(384)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("small")
        assert "MiniLM" in emb.model_name or "all-MiniLM" in emb.model_name

    def test_tier_medium_resolves_correct_model_name(self):
        mock_model = self._make_mock_model(768)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("medium")
        assert "bge" in emb.model_name or "BAAI" in emb.model_name

    def test_tier_large_resolves_correct_model_name(self):
        mock_model = self._make_mock_model(1024)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("large")
        assert "e5" in emb.model_name or "intfloat" in emb.model_name

    def test_dimension_from_registry(self):
        mock_model = self._make_mock_model(384)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("small")
        assert emb.dimension == 384

    def test_encode_returns_correct_shape(self):
        mock_model = self._make_mock_model(384)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("small")
        out = emb.encode(["hello", "world"])
        assert out.shape == (2, 384)
        assert out.dtype == np.float32

    def test_encode_single_returns_correct_shape(self):
        mock_model = self._make_mock_model(384)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("small")
        out = emb.encode_single("hello")
        assert out.shape == (384,)

    def test_literal_model_name_uses_model_dimension(self):
        """When a full model ID is given, dimension is discovered from model."""
        mock_model = self._make_mock_model(256)
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("some/custom-model")
        assert emb.dimension == 256

    def test_e5_prefix_applied(self):
        """Large (e5) tier prepends 'query: ' to texts."""
        mock_model = self._make_mock_model(1024)
        captured = []

        def capture_encode(texts, **kwargs):
            captured.extend(texts)
            n = len(texts)
            vecs = np.ones((n, 1024), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

        mock_model.encode = capture_encode
        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer", return_value=mock_model):
            from tadbench.embedders.sentence_transformer import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder("large")
        emb.encode(["hello", "world"])
        assert all(t.startswith("query: ") for t in captured)
