"""sentence-transformers backend (``pip install text2ipc[st]``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import normalize

#: Query / passage prefixes some model families were trained with.
_PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "bge-.*-en": ("Represent this sentence for searching relevant passages: ", ""),
}


def prefixes_for(model: str) -> tuple[str, str]:
    import re

    lowered = model.lower()
    for pattern, pair in _PREFIXES.items():
        if re.search(pattern, lowered):
            return pair
    return ("", "")


class SentenceTransformerEmbedder:
    def __init__(self, model: str, *, batch_size: int = 64, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the 'st' extra: uv add 'text2ipc[st]'") from e
        self._model_name = model
        self._model = SentenceTransformer(model, device=device)
        self._batch_size = batch_size
        self._query_prefix, self._passage_prefix = prefixes_for(model)

    @property
    def name(self) -> str:
        return f"st:{self._model_name}"

    @property
    def dim(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:  # older sentence-transformers
            getter = self._model.get_sentence_embedding_dimension
        return int(getter())

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(
            [self._passage_prefix + t for t in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        )
        return normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages_raw([self._query_prefix + text])[0]

    def embed_passages_raw(self, texts: Sequence[str]) -> np.ndarray:
        out = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return normalize(out)
