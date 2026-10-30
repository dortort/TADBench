"""Technique 3: Intent-Content Consistency Detection (ICCD)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.base import BaseDetector
from tadbench.embedders.protocol import EmbedderProtocol
from tadbench.utils.chunker import get_chunker

logger = logging.getLogger(__name__)

ScorerType = Literal["bi_encoder", "cross_encoder"]
BodyStrategyType = Literal["full", "summary_first_512", "chunk_mean", "top_k_suspicious"]

_TOP_K_SUSPICIOUS = 3   # chunks used by top_k_suspicious strategy
_PREFLIGHT_THRESHOLD = 0.5  # fraction of empty descriptions that triggers ValueError


class IntentContentDetector(BaseDetector):
    """Detect threats by measuring semantic mismatch between a prompt's
    advertised description and its actual body.

    Bi-encoder mode:  score = 1.0 - cosine_similarity(embed(desc), embed(body))
    Cross-encoder mode: score = P(contradiction) from an NLI 3-class model.

    Higher scores indicate greater inconsistency and thus more likely malicious.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        scorer: ScorerType = "bi_encoder",
        body_strategy: BodyStrategyType = "full",
        cross_encoder_model: str = "cross-encoder/nli-deberta-v3-base",
        **kwargs: object,
    ) -> None:
        super().__init__(embedder, **kwargs)
        self.scorer = scorer
        self.body_strategy = body_strategy
        self.cross_encoder_model_name = cross_encoder_model

        self.cross_encoder_calls_total = 0
        self.cross_encoder_input_words = 0
        self._cross_encoder = None  # lazy-loaded
        self._chunker = get_chunker("sentence")

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"intent_content_{self.scorer}_{self.body_strategy}"

    @property
    def embedding_calls_per_sample(self) -> int:
        if self.scorer == "cross_encoder":
            return 0
        # bi_encoder: description + body (body may add more calls if chunked)
        if self.body_strategy in ("chunk_mean", "top_k_suspicious"):
            return 2  # conservative estimate; actual varies with chunk count
        return 2  # description + body

    @property
    def cross_encoder_calls_per_sample(self) -> int:
        return 1 if self.scorer == "cross_encoder" else 0

    def fit(self, samples: list[BenchmarkSample]) -> None:
        """Lazy-load cross-encoder model if scorer='cross_encoder'."""
        if self.scorer == "cross_encoder":
            self._load_cross_encoder()

    def score(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Score all samples. Returns NaN for samples without descriptions."""
        if not samples:
            return np.array([], dtype=float)

        # Pre-flight: raise if >50% of samples have empty descriptions
        empty_count = sum(1 for s in samples if not s.advertised_description.strip())
        if empty_count / len(samples) > _PREFLIGHT_THRESHOLD:
            raise ValueError(
                f"{empty_count}/{len(samples)} samples have empty advertised_description "
                f"({empty_count / len(samples):.0%} > {_PREFLIGHT_THRESHOLD:.0%}). "
                "IntentContentDetector requires descriptions. "
                "Use a compatible dataset (spml, synthetic_mismatch) or "
                "check TECHNIQUE_DATASET_COMPAT in tadbench.config."
            )

        if self.scorer == "bi_encoder":
            return self._score_bi_encoder(samples)
        else:
            if self._cross_encoder is None:
                self._load_cross_encoder()
            return self._score_cross_encoder(samples)

    def score_single(self, sample: BenchmarkSample) -> float:
        """Score a single sample (used for latency profiling)."""
        result = self.score([sample])
        return float(result[0])

    # ------------------------------------------------------------------
    # Scoring implementations
    # ------------------------------------------------------------------

    def _score_bi_encoder(self, samples: list[BenchmarkSample]) -> np.ndarray:
        scores = np.full(len(samples), np.nan, dtype=float)

        for i, sample in enumerate(samples):
            if not sample.advertised_description.strip():
                logger.warning("Sample %s has empty description; returning NaN.", sample.id)
                continue

            desc_emb = self.embedder.encode_single(sample.advertised_description)
            body_emb = self._embed_body(sample)

            # Embedder normalises to unit vectors; dot product == cosine similarity
            cos_sim = float(np.dot(desc_emb, body_emb))
            # Clamp to [-1, 1] to handle floating-point drift
            cos_sim = max(-1.0, min(1.0, cos_sim))
            scores[i] = 1.0 - cos_sim  # higher = more dissimilar = more suspicious

        return scores

    def _score_cross_encoder(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Score via NLI cross-encoder; returns P(contradiction) per sample."""
        import scipy.special as sp  # softmax

        scores = np.full(len(samples), np.nan, dtype=float)

        for i, sample in enumerate(samples):
            if not sample.advertised_description.strip():
                logger.warning("Sample %s has empty description; returning NaN.", sample.id)
                continue

            body_text = self._get_body_text(sample)
            pair = [sample.advertised_description, body_text]

            # CrossEncoder.predict returns raw logits; shape (3,) for NLI
            logits = self._cross_encoder.predict([pair])[0]
            self.cross_encoder_calls_total += 1
            self.cross_encoder_input_words += sum(len(text.split()) for text in pair)
            probs = sp.softmax(logits)

            # NLI label order for nli-deberta-v3-base: [contradiction, neutral, entailment]
            # (verified via model.config.label2id; assumed fixed for this model family)
            contradiction_prob = float(probs[0])
            scores[i] = contradiction_prob

        return scores

    # ------------------------------------------------------------------
    # Body embedding helpers
    # ------------------------------------------------------------------

    def _embed_body(self, sample: BenchmarkSample) -> np.ndarray:
        """Return a single embedding vector representing the prompt body."""
        body_text = self._get_body_text(sample)

        if self.body_strategy in ("full", "summary_first_512"):
            return self.embedder.encode_single(body_text)

        elif self.body_strategy == "chunk_mean":
            chunks = self._chunker.chunk(body_text)
            if not chunks:
                return self.embedder.encode_single(body_text)
            chunk_embs = self.embedder.encode(chunks)  # (n_chunks, dim)
            mean_emb = chunk_embs.mean(axis=0)
            # Re-normalise to unit vector
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-10:
                mean_emb = mean_emb / norm
            return mean_emb

        elif self.body_strategy == "top_k_suspicious":
            chunks = self._chunker.chunk(body_text)
            if not chunks:
                return self.embedder.encode_single(body_text)
            chunk_embs = self.embedder.encode(chunks)  # (n_chunks, dim)
            centroid = chunk_embs.mean(axis=0)
            distances = np.linalg.norm(chunk_embs - centroid, axis=1)
            k = min(_TOP_K_SUSPICIOUS, len(chunks))
            top_indices = np.argsort(distances)[-k:]
            suspicious_embs = chunk_embs[top_indices]  # (k, dim)
            mean_emb = suspicious_embs.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-10:
                mean_emb = mean_emb / norm
            return mean_emb

        else:
            raise ValueError(f"Unknown body_strategy: {self.body_strategy!r}")

    def _get_body_text(self, sample: BenchmarkSample) -> str:
        """Apply body_strategy pre-processing that works at the text level."""
        if self.body_strategy == "summary_first_512":
            tokens = sample.prompt_body.split()
            return " ".join(tokens[:512])
        # All other strategies use the full text (chunking is done in _embed_body)
        return sample.prompt_body

    # ------------------------------------------------------------------
    # Cross-encoder loading
    # ------------------------------------------------------------------

    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            logger.info("Loaded cross-encoder: %s", self.cross_encoder_model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is required for cross_encoder scorer. "
                "Install it with: pip install sentence-transformers"
            ) from exc
