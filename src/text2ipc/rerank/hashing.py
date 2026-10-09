"""Deterministic reranker for tests: bag-of-words cosine between query and text,
stretched to a logit-like range so that ``sigmoid`` spans 0..1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..embeddings.hashing import HashEmbedder


class HashReranker:
    def __init__(self, dim: int = 64):
        self._embedder = HashEmbedder(dim)

    @property
    def name(self) -> str:
        return f"hash:{self._embedder.dim}"

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        q = self._embedder.embed_query(query)
        cos = self._embedder.embed_passages(texts) @ q
        return (8.0 * (cos - 0.5)).astype(np.float32)
