"""sentence-transformers CrossEncoder backend (``pip install text2ipc[st]``).

The CrossEncoder API was renamed in sentence-transformers 4.0 (``automodel_args`` to
``model_kwargs``, ``activation_fct`` to ``activation_fn``); both generations are
supported by reading the installed signatures once.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import numpy as np


def _pick(signature: inspect.Signature, *names: str) -> str | None:
    """The first of ``names`` the callable accepts, or None."""
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return names[0]
    return next((n for n in names if n in params), None)


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

        init_kw = _pick(inspect.signature(CrossEncoder.__init__), "model_kwargs", "automodel_args")
        self._activation_kw = _pick(
            inspect.signature(CrossEncoder.predict), "activation_fn", "activation_fct"
        )
        if init_kw is None or self._activation_kw is None:
            raise ImportError(
                "This sentence-transformers version has an unknown CrossEncoder API; "
                "install sentence-transformers>=3.0"
            )
        kwargs = {"device": device, "max_length": max_length, init_kw: {"torch_dtype": dtype}}
        self._model = CrossEncoder(model, **kwargs)

    @property
    def name(self) -> str:
        return f"ce:{self._model_name}"

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = self._model.predict(
            [(query, t) for t in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            **{self._activation_kw: self._identity},  # raw logits, not the default sigmoid
        )
        return np.asarray(out, dtype=np.float32).reshape(-1)
