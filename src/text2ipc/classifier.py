"""The simple interface: version + level + text -> ranked IPC symbols."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

from .config import LEVELS, default_model, home
from .embeddings import Embedder, get_embedder, model_slug
from .index import IpcIndex, available_indexes, scheme_table_path
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
    ) -> list[Match]:
        if normalize:
            text = normalize_query(text)
        params = SearchParams(
            level=level,
            top_k=top_k,
            gap=gap,
            weights=weights or Weights(),
            beam=beam or Beam(),
            auto_margin=auto_margin,
        )
        return search(self.index, self.embedder.embed_query(text), params)


def normalize_query(text: str) -> str:
    """Collapse whitespace and lower-case text that is mostly upper case.

    Patent titles are printed in capitals; the tokenizers of multilingual models
    handle them badly (subclass hit@10 on RPI titles went from 25% to 35% by
    lower-casing). Mixed-case abstracts are left alone.
    """
    return lowercase_if_shouting(collapse_whitespace(text))


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
