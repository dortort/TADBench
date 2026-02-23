"""Text chunking utilities for Technique 2 (Intra-Prompt Semantic Outlier Detection).

Two chunkers are provided:

* :class:`SentenceChunker` — splits on sentence boundaries via NLTK, with
  graceful fallbacks for text that cannot be sentence-tokenised.
* :class:`FixedTokenChunker` — splits on whitespace at ~128-token intervals,
  guaranteeing a minimum chunk size of 10 tokens.

Both expose a single ``chunk(text) -> list[str]`` method.

Factory
-------
Use :func:`get_chunker` to obtain a chunker by name::

    chunker = get_chunker("sentence")
    chunks = chunker.chunk("Some long prompt …")
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK punkt data bootstrap (downloaded once, silently)
# ---------------------------------------------------------------------------

def _ensure_punkt() -> None:
    """Download NLTK punkt_tab data on first use if not already present."""
    import nltk  # imported lazily so nltk is not required at module import time

    nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# FixedTokenChunker
# ---------------------------------------------------------------------------

class FixedTokenChunker:
    """Splits text into chunks of approximately *chunk_size* whitespace tokens.

    Parameters
    ----------
    chunk_size:
        Target number of whitespace-delimited tokens per chunk (default 128).
    min_tokens:
        Minimum tokens required for a chunk to be kept (default 10).
        Trailing tokens that form a chunk smaller than this threshold are
        merged into the preceding chunk.
    """

    def __init__(self, chunk_size: int = 128, min_tokens: int = 10) -> None:
        self.chunk_size = chunk_size
        self.min_tokens = min_tokens

    def chunk(self, text: str) -> list[str]:
        """Split *text* into fixed-size token chunks.

        Returns
        -------
        list[str]
            Non-empty list of chunk strings.  The original text is always
            returned as a single-element list when it is too short to split.
        """
        tokens = text.split()
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            if len(chunk_tokens) < self.min_tokens and chunks:
                # Merge tiny trailing fragment into the last chunk
                last = chunks.pop()
                chunks.append(last + " " + " ".join(chunk_tokens))
            else:
                chunks.append(" ".join(chunk_tokens))
            start = end

        return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class SentenceChunker:
    """Splits text into sentence-level chunks using NLTK ``sent_tokenize``.

    Fallback chain when sentence tokenisation produces only one chunk:

    1. Split on ``\\n`` (newlines).
    2. Delegate to :class:`FixedTokenChunker` (128-token windows).

    Parameters
    ----------
    fixed_chunk_size:
        ``chunk_size`` passed to the internal :class:`FixedTokenChunker`
        fallback.
    """

    def __init__(self, fixed_chunk_size: int = 128) -> None:
        self._fixed_chunker = FixedTokenChunker(chunk_size=fixed_chunk_size)

    def chunk(self, text: str) -> list[str]:
        """Chunk *text* into sentences with graceful fallbacks.

        Returns
        -------
        list[str]
            Non-empty list of chunk strings.
        """
        import nltk  # noqa: PLC0415

        _ensure_punkt()

        # Primary: NLTK sentence tokenisation
        sentences: list[str] = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 1:
            return sentences

        # Fallback 1: newline splitting
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines

        # Fallback 2: fixed-token chunker
        fixed_chunks = self._fixed_chunker.chunk(text)
        if len(fixed_chunks) > 1:
            return fixed_chunks

        # Last resort: return the whole text as one chunk
        return [text]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_chunker(name: str) -> "SentenceChunker | FixedTokenChunker":
    """Return a chunker instance by name.

    Parameters
    ----------
    name:
        ``"sentence"`` for :class:`SentenceChunker`, or ``"fixed_128"`` for
        :class:`FixedTokenChunker` with 128-token windows.

    Raises
    ------
    ValueError
        If *name* is not a recognised chunker identifier.
    """
    _registry: dict[str, Callable[[], SentenceChunker | FixedTokenChunker]] = {
        "sentence": SentenceChunker,
        "fixed_128": lambda: FixedTokenChunker(chunk_size=128),
    }
    if name not in _registry:
        raise ValueError(
            f"Unknown chunker {name!r}. Available: {sorted(_registry)}"
        )
    return _registry[name]()
