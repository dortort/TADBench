"""Technique 1: Threat-Prototype Similarity (TPS) detector."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from tadbench.data.schema import BenchmarkSample
from tadbench.detectors.base import BaseDetector
from tadbench.embedders.protocol import EmbedderProtocol
from tadbench.utils.io import read_jsonl

logger = logging.getLogger(__name__)

IndexType = Literal["flat", "hnsw", "ivf"]
AggregationType = Literal["max", "mean", "weighted"]


class ThreatPrototypeDetector(BaseDetector):
    """Detect threats by cosine similarity to a bank of known malicious prompts.

    Builds a FAISS index over threat-bank embeddings during fit(). At score
    time each query prompt is embedded and its top-k nearest neighbours in the
    threat bank are retrieved; the neighbour similarities are aggregated into a
    single anomaly score (higher = more likely malicious).
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        index_type: IndexType = "flat",
        top_k: int = 1,
        aggregation: AggregationType = "max",
        **kwargs: object,
    ) -> None:
        super().__init__(embedder, **kwargs)
        self.index_type = index_type
        self.top_k = top_k
        self.aggregation = aggregation

        self._index: faiss.Index | None = None
        self._sample_count: int = 0
        self._metadata: dict = {}

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"threat_prototype_{self.index_type}_top{self.top_k}_{self.aggregation}"

    def fit(self, samples: list[BenchmarkSample]) -> None:
        """Embed all samples' prompt_body texts and build a FAISS index."""
        if not samples:
            raise ValueError("fit() called with empty sample list.")

        texts = [s.prompt_body for s in samples]
        embeddings = self.embedder.encode(texts)  # (n, dim)
        self._build_faiss_index(embeddings)
        self._sample_count = len(samples)

    def score(self, samples: list[BenchmarkSample]) -> np.ndarray:
        """Return similarity scores, shape (n,). Higher = more malicious."""
        if self._index is None:
            raise RuntimeError("Detector not fitted. Call fit() or load_index() first.")

        texts = [s.prompt_body for s in samples]
        embeddings = self.embedder.encode(texts)  # (n, dim)
        return self._query_index(embeddings)

    def score_single(self, sample: BenchmarkSample) -> float:
        """Score a single sample (used for latency profiling)."""
        return float(self.score([sample])[0])

    # ------------------------------------------------------------------
    # Index construction helpers
    # ------------------------------------------------------------------

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build the FAISS index from a float32 embedding matrix."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n, dim = embeddings.shape

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif self.index_type == "hnsw":
            # The default metric is L2 distance (lower = closer); _query_index
            # treats results as similarities, so inner product is required.
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)  # M=32
        elif self.index_type == "ivf":
            nlist = min(100, max(1, n // 10))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type!r}")

        index.add(embeddings)
        self._index = index

    def _query_index(self, embeddings: np.ndarray) -> np.ndarray:
        """Query the index and aggregate top-k similarities."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        k = min(self.top_k, self._index.ntotal)

        similarities, _ = self._index.search(embeddings, k)  # (n, k)

        if self.aggregation == "max":
            return similarities[:, 0].astype(float)
        elif self.aggregation == "mean":
            return similarities.mean(axis=1).astype(float)
        elif self.aggregation == "weighted":
            # Distance-weighted mean: weight[i] = 1/(rank+1)
            weights = 1.0 / (np.arange(k, dtype=float) + 1.0)
            weights /= weights.sum()
            return (similarities * weights).sum(axis=1).astype(float)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def build_index(self, threat_bank_path: str | Path) -> None:
        """Load a JSONL threat bank, embed all entries, and build the index.

        Also stores metadata that can be serialised alongside the index.
        """
        threat_bank_path = Path(threat_bank_path)
        raw = read_jsonl(threat_bank_path)
        texts = [r["prompt_body"] for r in raw]
        if not texts:
            raise ValueError(f"Threat bank at {threat_bank_path} is empty.")

        bank_hash = _file_sha256(threat_bank_path)
        embeddings = self.embedder.encode(texts)
        self._build_faiss_index(embeddings)
        self._sample_count = len(texts)

        self._metadata = {
            "embedder": self.embedder.model_name,
            "index_type": self.index_type,
            "threat_bank_hash": f"sha256:{bank_hash}",
            "sample_count": self._sample_count,
            "dimension": self.embedder.dimension,
            "build_timestamp": _utc_iso(),
        }

    def save_index(self, index_path: str | Path) -> None:
        """Write the FAISS index and companion metadata JSON to disk."""
        if self._index is None:
            raise RuntimeError("No index built yet. Call fit() or build_index() first.")

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

        meta_path = index_path.with_suffix("").with_suffix(".meta.json")
        meta_path.write_text(json.dumps(self._metadata, indent=2))
        logger.info("Saved index to %s and metadata to %s", index_path, meta_path)

    def load_index(self, index_path: str | Path, threat_bank_path: str | Path | None = None) -> None:
        """Load a pre-built FAISS index from disk.

        Optionally checks staleness against a threat bank file.
        """
        index_path = Path(index_path)
        self._index = faiss.read_index(str(index_path))

        meta_path = index_path.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            self._metadata = json.loads(meta_path.read_text())
            self._sample_count = self._metadata.get("sample_count", self._index.ntotal)

            if threat_bank_path is not None:
                current_hash = f"sha256:{_file_sha256(Path(threat_bank_path))}"
                stored_hash = self._metadata.get("threat_bank_hash", "")
                if current_hash != stored_hash:
                    logger.warning(
                        "Loaded index may be stale: threat bank hash mismatch "
                        "(stored=%s, current=%s). Rebuild with build_index().",
                        stored_hash,
                        current_hash,
                    )
        else:
            logger.warning("No metadata file found alongside %s.", index_path)
            self._sample_count = self._index.ntotal

        logger.info("Loaded index with %d vectors from %s", self._index.ntotal, index_path)


# ------------------------------------------------------------------
# Private utilities
# ------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
