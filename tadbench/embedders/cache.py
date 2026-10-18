"""EmbeddingCache: a caching wrapper around any EmbedderProtocol.

Cache keys are ``sha256(model_name + text)`` truncated to 16 hex characters.
Embeddings are stored in an in-memory dict and periodically persisted to an
``.npz`` file on disk (no h5py dependency).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from tadbench.embedders.protocol import EmbedderProtocol

logger = logging.getLogger(__name__)

# Flush to disk after this many *new* embeddings have been added since the
# last save.
_SAVE_INTERVAL = 1000


def _cache_key(model_name: str, text: str) -> str:
    """Return a 16-hex-char cache key for *(model_name, text)*."""
    raw = (model_name + text).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class EmbeddingCache(EmbedderProtocol):
    """Caching wrapper for any :class:`EmbedderProtocol` implementation.

    Embeddings are stored in an in-memory dictionary and optionally persisted
    to ``<cache_dir>/<model_hash>.npz`` so they survive across process
    invocations.

    Parameters
    ----------
    embedder:
        The underlying embedder to call on cache misses.
    cache_dir:
        Directory where cache ``.npz`` files are stored.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        cache_dir: str = "indexes/cache",
    ) -> None:
        self._embedder = embedder
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory store: key -> np.ndarray of shape (dim,)
        self._store: dict[str, np.ndarray] = {}

        # Hit-rate tracking
        self._hits: int = 0
        self._total: int = 0

        # Count of new embeddings added since last flush
        self._new_since_save: int = 0

        self._bypassed: bool = False

        # Load any existing cache from disk
        self._load()

    # ------------------------------------------------------------------
    # EmbedderProtocol implementation
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for *texts*, using cached results where available.

        Cache misses are batched and forwarded to the wrapped embedder.
        """
        if self._bypassed:
            return np.asarray(self._embedder.encode(texts), dtype=np.float32)

        keys = [_cache_key(self._embedder.model_name, t) for t in texts]
        self._total += len(texts)

        # Identify misses — deduplicate so each unique missing key is embedded once.
        # Keys that appear multiple times in a single call are counted as hits after
        # the first occurrence.
        seen_in_call: set[str] = set()
        unique_miss_keys: list[str] = []   # ordered, deduped miss keys
        unique_miss_texts: list[str] = []  # corresponding texts

        for i, key in enumerate(keys):
            if key in self._store or key in seen_in_call:
                self._hits += 1
            else:
                seen_in_call.add(key)
                unique_miss_keys.append(key)
                unique_miss_texts.append(texts[i])
        # Embed unique misses in one batch
        if unique_miss_texts:
            miss_embeddings = self._embedder.encode(unique_miss_texts)
            for local_i, key in enumerate(unique_miss_keys):
                self._store[key] = miss_embeddings[local_i]
            self._new_since_save += len(unique_miss_texts)

        # Assemble result in original order
        result = np.stack([self._store[k] for k in keys], axis=0)

        # Periodic persistence
        if self._new_since_save >= _SAVE_INTERVAL:
            self._save()

        return result.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Return the embedding for a single *text*."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @property
    def bypassed(self) -> bool:
        """Whether :meth:`encode` is currently skipping the cache."""
        return self._bypassed

    @contextmanager
    def bypass(self) -> Iterator[None]:
        """Send every ``encode`` call to the wrapped embedder within the block.

        The store and hit-rate counters are neither read nor updated, so
        latency measured inside the block reflects real embedding cost.
        """
        previous = self._bypassed
        self._bypassed = True
        try:
            yield
        finally:
            self._bypassed = previous

    @property
    def hit_rate(self) -> float:
        """Fraction of encode requests served from cache (0.0 if no calls yet)."""
        if self._total == 0:
            return 0.0
        return self._hits / self._total

    def clear(self) -> None:
        """Remove all cached embeddings from memory and delete the disk file."""
        self._store.clear()
        self._hits = 0
        self._total = 0
        self._new_since_save = 0
        cache_file = self._cache_file_path()
        if cache_file.exists():
            cache_file.unlink()
        logger.debug("EmbeddingCache cleared (model=%s)", self._embedder.model_name)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _cache_file_path(self) -> Path:
        # Use a hash of the model name so different models get separate files.
        model_hash = hashlib.sha256(self._embedder.model_name.encode()).hexdigest()[:16]
        return self._cache_dir / f"{model_hash}.npz"

    def _save(self) -> None:
        """Persist in-memory cache to disk as a compressed .npz file."""
        if not self._store:
            return
        path = self._cache_file_path()
        # npz requires string keys; the 16-char hex keys are valid.
        np.savez_compressed(str(path), **self._store)
        self._new_since_save = 0
        logger.debug(
            "EmbeddingCache saved %d entries to %s", len(self._store), path
        )

    def _load(self) -> None:
        """Load cache from disk if a persisted file exists."""
        path = self._cache_file_path()
        if not path.exists():
            return
        try:
            data = np.load(str(path), allow_pickle=False)
            self._store = {k: data[k] for k in data.files}
            logger.debug(
                "EmbeddingCache loaded %d entries from %s",
                len(self._store),
                path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load embedding cache from %s: %s. Starting fresh.",
                path,
                exc,
            )
            self._store = {}

    def __del__(self) -> None:
        """Flush any unsaved embeddings when the object is garbage-collected."""
        if self._new_since_save > 0:
            try:
                self._save()
            except Exception:  # noqa: BLE001
                pass
