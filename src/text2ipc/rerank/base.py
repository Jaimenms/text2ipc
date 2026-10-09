"""Second stage: a cross-encoder judges every (text, entry path) pair of the first
stage's candidates and the two scores are fused.

A reranker is selected by a spec string ``<backend>:<model>``:

- ``ce:BAAI/bge-reranker-v2-m3``  sentence-transformers CrossEncoder (default)
- ``hash:64``                      bag-of-words overlap, for tests only

Measured on the evals (docs/evals.md): over the top-50 of e5-base on the Portuguese
scheme, bge-reranker-v2-m3 lifts subclass@1 from 23.6% to 32.3%.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

FUSIONS = ("blend", "judge", "product")


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str:
        """Full spec string."""

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """One relevance logit per text, higher is better (``sigmoid`` maps to 0..1)."""


def get_reranker(spec: str | Reranker, **kwargs) -> Reranker:
    if not isinstance(spec, str):
        return spec
    backend, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Reranker spec must look like 'backend:model', got {spec!r}")
    if backend == "ce":
        from .st import CrossEncoderReranker

        return CrossEncoderReranker(model, **kwargs)
    if backend == "hash":
        from .hashing import HashReranker

        return HashReranker(int(model))
    raise ValueError(f"Unknown reranker backend {backend!r}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def fuse(scores: np.ndarray, logits: np.ndarray, how: str = "blend") -> np.ndarray:
    """Combine first-stage scores with the judge's logits.

    - ``blend``: ``score * (0.5 + 0.5 * sigmoid(logit))``, the default: the judge
      can halve a score but the cosine still orders ties, and the result stays
      comparable with unreranked scores.
    - ``judge``: ``sigmoid(logit)`` alone.
    - ``product``: ``score * sigmoid(logit)``.
    """
    s = np.asarray(scores, dtype=np.float64)
    j = sigmoid(logits)
    if how == "blend":
        return s * (0.5 + 0.5 * j)
    if how == "judge":
        return j
    if how == "product":
        return s * j
    raise ValueError(f"fusion must be one of {FUSIONS}, got {how!r}")
