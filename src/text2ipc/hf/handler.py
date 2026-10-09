"""Hugging Face Inference Endpoints handler: text in, ranked IPC symbols out.

Copied to the root of the model repository by ``t2ipc hf-export``. The repository
also carries ``scheme/`` and ``index/`` (parquet), ``text2ipc.json`` (which index to
serve) and a copy of the ``text2ipc`` package, so no PyPI release is required.

Request::

    {"inputs": "enxada manual com duas lâminas",
     "parameters": {"level": "group", "top_k": 5}}

Response (one list per input; a single string yields a single list)::

    [{"symbol": "A01B 1/10", "canonical": "A01B0001100000", "level": "subgroup",
      "score": 0.83, "similarity": 0.81, "title": "...", "path": "..."}, ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_PARAMS = {
    "level", "top_k", "gap", "auto_margin", "normalize", "chunking", "rerank", "candidates", "fusion"
}


class EndpointHandler:
    def __init__(self, path: str = ""):
        root = Path(path or ".").resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))  # vendored text2ipc package
        from text2ipc.classifier import IpcClassifier

        cfg = json.loads((root / "text2ipc.json").read_text())
        self.clf = IpcClassifier(
            cfg["version"],
            lang=cfg["lang"],
            model=cfg.get("embedder") or cfg["model"],
            root=root,
        )
        _ = self.clf.index  # load eagerly so the first request is fast
        _ = self.clf.embedder

    def __call__(self, data: dict[str, Any]) -> list[dict] | list[list[dict]]:
        inputs = data.get("inputs", "")
        params = {k: v for k, v in (data.get("parameters") or {}).items() if k in ALLOWED_PARAMS}
        if isinstance(inputs, str):
            return self._one(inputs, params)
        return [self._one(text, params) for text in inputs]

    def _one(self, text: str, params: dict[str, Any]) -> list[dict]:
        return [
            {
                "symbol": m.pretty,
                "canonical": m.symbol,
                "level": m.level,
                "depth": m.depth,
                "score": round(m.score, 4),
                "similarity": round(m.similarity, 4),
                "judge": None if m.judge is None else round(m.judge, 4),
                "title": m.title,
                "path": m.text,
            }
            for m in self.clf.classify(text, **params)
        ]
