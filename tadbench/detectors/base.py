"""Base detector ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tadbench.embedders.protocol import EmbedderProtocol
from tadbench.data.schema import BenchmarkSample


class BaseDetector(ABC):
    """Abstract base class for all threat detection techniques.

    Stores the embedder at construction time. Subclasses implement
    fit(), score(), score_single(), and the name property.

    Score convention: higher values indicate a sample is more likely malicious.
    """

    def __init__(self, embedder: EmbedderProtocol, **kwargs: object) -> None:
        self.embedder = embedder

    @abstractmethod
    def fit(self, samples: list[BenchmarkSample]) -> None:
        """Fit detector on training data (e.g., build index, fit outlier model)."""

    @abstractmethod
    def score(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Score samples. Returns array of shape (n,). Higher = more likely malicious."""

    @abstractmethod
    def score_single(self, sample: BenchmarkSample) -> float:
        """Score a single sample. Used for latency profiling."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique detector name (used in result reporting)."""

    @property
    def embedding_calls_per_sample(self) -> int:
        """Number of embedding encode calls per sample (for cost proxy reporting)."""
        return 1

    @property
    def cross_encoder_calls_per_sample(self) -> int:
        """Number of cross-encoder calls per sample (for cost proxy reporting)."""
        return 0
