"""EmbedderProtocol: abstract base class defining the embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbedderProtocol(ABC):
    """Abstract base class for all embedding backends.

    All implementations must return L2-normalised vectors so that inner
    products equal cosine similarities.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(n, dim)`` where ``n = len(texts)``.
        """

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Parameters
        ----------
        text:
            String to embed.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(dim,)``.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Fully-qualified model identifier (e.g. HuggingFace repo name)."""
