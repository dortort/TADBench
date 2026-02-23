"""SentenceTransformerEmbedder: concrete EmbedderProtocol backed by sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from tadbench.config import EMBEDDER_REGISTRY
from tadbench.embedders.protocol import EmbedderProtocol


class SentenceTransformerEmbedder(EmbedderProtocol):
    """Embedding backend using the ``sentence-transformers`` library.

    Parameters
    ----------
    model_name_or_tier:
        Either a tier shorthand (``"small"``, ``"medium"``, ``"large"``) that is
        resolved via :data:`tadbench.config.EMBEDDER_REGISTRY`, or a full
        HuggingFace model identifier such as
        ``"sentence-transformers/all-MiniLM-L6-v2"``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).
    batch_size:
        Number of texts per encoding batch passed to the underlying model.
    """

    def __init__(
        self,
        model_name_or_tier: str,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        # Resolve tier shorthand
        if model_name_or_tier in EMBEDDER_REGISTRY:
            reg = EMBEDDER_REGISTRY[model_name_or_tier]
            self._model_name: str = reg["model"]
            self._dim: int = reg["dim"]
            self._prefix: dict[str, str] | None = reg.get("prefix")
        else:
            # Treat the argument as a literal model identifier; dimension will
            # be discovered from the loaded model.
            self._model_name = model_name_or_tier
            self._dim = 0  # set after model load below
            self._prefix = None

        self.batch_size = batch_size
        self._model = SentenceTransformer(self._model_name, device=device)

        # If dim was not known from the registry, infer it now.
        if self._dim == 0:
            self._dim = self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # EmbedderProtocol implementation
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return an ``(n, dim)`` float32 array.

        Texts are prefixed with ``"query: "`` when the backing model is
        ``intfloat/e5-large-v2`` (or any model whose registry entry has a
        ``prefix`` dict with a ``"query"`` key).
        """
        prepared = self._apply_prefix(texts)
        embeddings: np.ndarray = self._model.encode(
            prepared,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text and return a ``(dim,)`` float32 array."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_prefix(self, texts: list[str]) -> list[str]:
        """Prepend query prefix for models that require it (e.g. e5-large)."""
        if self._prefix is None:
            return texts
        query_prefix = self._prefix.get("query", "")
        if not query_prefix:
            return texts
        return [query_prefix + t for t in texts]
