"""sentence-transformers CrossEncoder backend (``pip install text2ipc[st]``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class CrossEncoderReranker:
    def __init__(
        self,
        model: str,
        *,
        max_length: int = 768,
        batch_size: int = 32,
        device: str | None = None,
    ):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install the 'st' extra: uv add 'text2ipc[st]'") from e
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self._model_name = model
        self._batch_size = batch_size
        self._identity = torch.nn.Identity()
        self._model = CrossEncoder(
            model, device=device, max_length=max_length, model_kwargs={"torch_dtype": dtype}
        )

    @property
    def name(self) -> str:
        return f"ce:{self._model_name}"

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = self._model.predict(
            [(query, t) for t in texts],
            batch_size=self._batch_size,
            activation_fn=self._identity,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32).reshape(-1)
