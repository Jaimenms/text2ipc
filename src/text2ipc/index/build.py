"""Build an index for one scheme version, reusing vectors from an older one.

A node's vector depends only on its rendered text (the full path of titles) and the
model. For every symbol present in the previous index with an identical text hash the
old vector is copied instead of recomputed. Between two yearly IPC versions that is
typically well over 95 percent of the entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..embeddings.base import Embedder
from ..scheme.table import SchemeTable
from .store import IndexMeta, IpcIndex


@dataclass
class BuildReport:
    version: str
    lang: str
    model: str
    total: int = 0
    reused: int = 0
    computed: int = 0
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    previous: str | None = None

    def summary(self) -> str:
        head = f"{self.version} {self.lang} {self.model}: {self.total} entries"
        if self.previous is None:
            return f"{head}, all {self.computed} embedded from scratch"
        return (
            f"{head}; reused {self.reused}, embedded {self.computed} "
            f"(added {len(self.added)}, changed {len(self.changed)}, "
            f"removed {len(self.removed)}) vs {self.previous}"
        )


def build_index(
    scheme: SchemeTable,
    embedder: Embedder,
    *,
    version: str,
    lang: str,
    previous: IpcIndex | tuple[Path, Path] | None = None,
    titles_overlay: str | None = None,
    batch_size: int = 256,
    progress=None,
) -> tuple[IpcIndex, BuildReport]:
    """``previous`` is an index or a ``(index parquet, scheme parquet)`` pair."""
    lang = lang.upper()
    report = BuildReport(version=version, lang=lang, model=embedder.name, total=len(scheme))

    prev = _load_previous(previous, embedder, lang, report)
    prev_hash = prev.hashes() if prev else {}
    prev_symbols = set(prev_hash)
    texts = scheme.texts()
    hashes = scheme.hashes()

    vectors = np.zeros((len(scheme), embedder.dim), dtype=np.float32)
    todo: list[int] = []
    for i, node in enumerate(scheme.nodes):
        old = prev_hash.get(node.symbol)
        if old is not None and old == hashes[i]:
            vectors[i] = prev.vectors[prev.position[node.symbol]]  # type: ignore[union-attr]
            report.reused += 1
        else:
            todo.append(i)
            (report.changed if old is not None else report.added).append(node.symbol)
    report.removed = sorted(prev_symbols - set(scheme.position))
    report.computed = len(todo)

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors[batch] = embedder.embed_passages([texts[i] for i in batch])
        if progress is not None:
            progress(min(start + batch_size, len(todo)), len(todo))

    meta = IndexMeta.now(
        version=version,
        lang=lang,
        model=embedder.name,
        dim=embedder.dim,
        rows=len(scheme),
        titles_overlay=titles_overlay,
    )
    return IpcIndex(meta, scheme, vectors, hashes), report


def _load_previous(
    previous: IpcIndex | tuple[Path, Path] | None,
    embedder: Embedder,
    lang: str,
    report: BuildReport,
) -> IpcIndex | None:
    if previous is None:
        return None
    prev = IpcIndex.read(*previous) if isinstance(previous, tuple) else previous
    if prev.meta.model != embedder.name or prev.meta.dim != embedder.dim:
        raise ValueError(
            f"Previous index was built with {prev.meta.model} ({prev.meta.dim}d), "
            f"cannot reuse for {embedder.name} ({embedder.dim}d)"
        )
    if prev.meta.lang != lang:
        raise ValueError(f"Previous index is {prev.meta.lang}, cannot reuse for {lang}")
    report.previous = prev.meta.version
    return prev
