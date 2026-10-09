"""The simple interface: version + level + text -> ranked IPC symbols."""

from __future__ import annotations

import dataclasses
import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np

from .chunking import split_text
from .config import DEFAULT_RERANKER, LEVELS, default_model, home
from .embeddings import Embedder, get_embedder, model_slug
from .embeddings.base import normalize
from .index import IpcIndex, available_indexes, scheme_table_path
from .rerank import Reranker, fuse, get_reranker, sigmoid
from .search import Beam, Match, SearchParams, Weights, search
from .textnorm import collapse_whitespace, lowercase_if_shouting


class IpcClassifier:
    """Loads one built index and answers queries against it.

    Construction is cheap; the index, its scheme table and the embedding model load on
    first use.
    """

    def __init__(
        self,
        version: str = "latest",
        *,
        lang: str = "EN",
        model: str | Embedder | None = None,
        root: Path | None = None,
    ):
        self.root = root or home()
        self.lang = lang.upper()
        self._model_spec = model or default_model()
        self.version = resolve_built_version(version, self.lang, self._model_name, self.root)
        self._index: IpcIndex | None = None
        self._embedder: Embedder | None = None
        self._rerankers: dict[str, Reranker] = {}
        #: How many chunks the last ``embed_text`` call used (1 when the text fit).
        self.last_chunks = 1

    @property
    def _model_name(self) -> str:
        return self._model_spec if isinstance(self._model_spec, str) else self._model_spec.name

    @property
    def index(self) -> IpcIndex:
        if self._index is None:
            path = _index_for(self.version, self.lang, self._model_name, self.root)
            self._index = IpcIndex.read(path, scheme_table_path(self.version, self.lang, self.root))
        return self._index

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self._model_spec)
            if self._embedder.name != self.index.meta.model:
                raise ValueError(
                    f"Index built with {self.index.meta.model}, embedder is {self._embedder.name}"
                )
        return self._embedder

    def classify(
        self,
        text: str,
        level: str = "subgroup",
        top_k: int = 10,
        *,
        gap: float | None = None,
        weights: Weights | None = None,
        beam: Beam | None = None,
        auto_margin: float = 0.02,
        normalize: bool = True,
        chunking: str = "mean",
        rerank: bool | str | Reranker = False,
        candidates: int | None = None,
        fusion: str = "blend",
    ) -> list[Match]:
        """Rank IPC entries for a text of any length.

        ``rerank`` adds a second stage: the first stage keeps ``candidates`` entries
        (default ``max(5 * top_k, 25)``), a cross-encoder (``True`` for the default
        model, or a spec such as ``ce:BAAI/bge-reranker-v2-m3``) judges each one
        against the text, and ``fusion`` combines the two scores (``blend``,
        ``judge`` or ``product``, see ``rerank.fuse``). ``gap`` applies to the first
        stage. Each match then carries its ``judge`` verdict in 0..1.

        Paragraphs (blank-line separated, e.g. a title above an abstract) are embedded
        separately, and a paragraph over the embedder's limit is cut into sentence
        chunks. ``chunking`` says how the pieces combine: ``"mean"`` searches with
        their unit-length mean, ``"max"`` searches with every vector and scores an
        entry by its best piece, ``"truncate"`` embeds the text whole and lets the
        embedder cut it (the old behaviour).
        """
        if normalize:
            text = normalize_query(text)
        n_first = top_k if not rerank else (candidates or max(5 * top_k, 25))
        params = SearchParams(
            level=level,
            top_k=n_first,
            gap=gap,
            weights=weights or Weights(),
            beam=beam or Beam(),
            auto_margin=auto_margin,
        )
        matches = search(self.index, self.embed_text(text, chunking=chunking), params)
        if not rerank or not matches:
            return matches[:top_k]
        reranker = self.reranker(rerank)
        logits = reranker.score(text, [m.text for m in matches])
        fused = fuse(np.array([m.score for m in matches]), logits, fusion)
        verdict = sigmoid(logits)
        order = np.argsort(-fused, kind="stable")[:top_k]
        return [
            dataclasses.replace(matches[i], score=float(fused[i]), judge=float(verdict[i]))
            for i in order
        ]

    def reranker(self, spec: bool | str | Reranker = True) -> Reranker:
        """The reranker for a spec (``True`` means the default), loaded once."""
        if spec is True:
            spec = DEFAULT_RERANKER
        if not isinstance(spec, str):
            return spec
        if spec not in self._rerankers:
            self._rerankers[spec] = get_reranker(spec)
        return self._rerankers[spec]

    def embed_text(self, text: str, *, chunking: str = "mean") -> np.ndarray:
        """Query vector for a text of any length; a ``(chunks, dim)`` stack for ``"max"``.

        Sets ``last_chunks``. A single paragraph within the token limit is embedded
        whole.
        """
        if chunking not in ("mean", "max", "truncate"):
            raise ValueError("chunking must be 'mean', 'max' or 'truncate'")
        embedder = self.embedder
        chunks = (
            [text]
            if chunking == "truncate"
            else split_text(text, embedder.max_tokens, embedder.count_tokens)
        )
        self.last_chunks = len(chunks)
        if len(chunks) == 1:
            return embedder.embed_query(text)
        vectors = embedder.embed_queries(chunks)
        if chunking == "max":
            return vectors
        return normalize(vectors.mean(axis=0))[0]


def normalize_query(text: str) -> str:
    """Collapse whitespace and lower-case text that is mostly upper case, paragraph by
    paragraph, keeping blank lines as paragraph breaks.

    Patent titles are printed in capitals; the tokenizers of multilingual models
    handle them badly (subclass hit@10 on RPI titles went from 25% to 35% by
    lower-casing). Mixed-case abstracts are left alone. Per paragraph, so that a
    shouting title above a mixed-case abstract is still lower-cased.
    """
    return "\n\n".join(lowercase_if_shouting(p) for p in collapse_whitespace(text).split("\n\n"))


def classify(
    text: str,
    version: str = "latest",
    level: str = "subgroup",
    top_k: int = 10,
    *,
    lang: str = "EN",
    model: str | None = None,
    root: Path | None = None,
    **kwargs,
) -> list[Match]:
    """One-call form. Classifiers are cached per (version, lang, model)."""
    if level not in (*LEVELS, "auto"):
        raise ValueError(f"level must be one of {LEVELS} or 'auto'")
    clf = _cached(version, lang.upper(), model or default_model(), str(root) if root else None)
    return clf.classify(text, level=level, top_k=top_k, **kwargs)


@lru_cache(maxsize=8)
def _cached(version: str, lang: str, model: str, root: str | None) -> IpcClassifier:
    return IpcClassifier(version, lang=lang, model=model, root=Path(root) if root else None)


def resolve_built_version(spec: str, lang: str, model: str, root: Path) -> str:
    """Resolve ``latest`` / ``current`` against indexes already built for lang+model."""
    if spec.isdigit() and len(spec) == 8:
        return spec
    slug = model_slug(model)
    versions = sorted(
        r.version for r in available_indexes(root) if r.lang == lang and r.model_slug == slug
    )
    if spec == "current":
        today = dt.date.today().strftime("%Y%m%d")
        versions = [v for v in versions if v <= today]
    if not versions:
        raise FileNotFoundError(
            f"No index built for lang={lang} model={model} under {root}. "
            "Run: t2ipc build --version <YYYYMMDD> (or t2ipc download)"
        )
    return versions[-1]


def _index_for(version: str, lang: str, model: str, root: Path) -> Path:
    slug = model_slug(model)
    for r in available_indexes(root):
        if r.version == version and r.lang == lang and r.model_slug == slug:
            return r.path
    raise FileNotFoundError(
        f"No index for version={version} lang={lang} model={model} under {root}. "
        f"Run: t2ipc build --version {version} --lang {lang} --model {model}"
    )
