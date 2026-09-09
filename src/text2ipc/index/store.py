"""One Parquet index per (IPC version, scheme language, embedding model).

The index carries only what depends on the model: ``symbol``, ``path`` (the ``|``-joined
symbol chain the text was built from), ``text_hash`` and the vector as float32 columns
``e000 .. eNNN``. Titles live in the scheme table (``SchemeTable``), so the same text
is rendered for display and for hashing. Metadata (version, language, model,
dimension, build time) is stored in the Parquet schema metadata under ``text2ipc``.

Rows keep the scheme's depth-first order, so every parent precedes its children; the
search code relies on that.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import LEVELS
from ..scheme.model import SYMBOL_SEPARATOR, IpcNode, text_hash
from ..scheme.table import SchemeTable

META_KEY = b"text2ipc"


@dataclass(frozen=True)
class IndexMeta:
    version: str
    lang: str
    model: str
    dim: int
    rows: int
    built_at: str
    titles_overlay: str | None = None

    @classmethod
    def now(cls, **kwargs) -> IndexMeta:
        return cls(built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **kwargs)


def vector_columns(dim: int) -> list[str]:
    width = max(3, len(str(dim - 1)))
    return [f"e{i:0{width}d}" for i in range(dim)]


class IpcIndex:
    """Scheme table + aligned vectors, plus the derived hierarchy arrays."""

    def __init__(
        self,
        meta: IndexMeta,
        scheme: SchemeTable,
        vectors: np.ndarray,
        hashes: list[str] | None = None,
    ):
        if vectors.shape != (len(scheme), meta.dim):
            raise ValueError(f"vectors {vectors.shape} do not match {len(scheme)} x {meta.dim}")
        self.meta = meta
        self.scheme = scheme
        self.nodes: list[IpcNode] = scheme.nodes
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.texts = scheme.texts()
        self.hashes_list = hashes or [text_hash(t) for t in self.texts]
        self.symbols = [n.symbol for n in self.nodes]
        self.position = scheme.position
        self.levels = np.array([LEVELS.index(n.level) for n in self.nodes], dtype=np.int8)
        self.parent_idx = np.array(
            [self.position[n.parent] if n.parent else -1 for n in self.nodes], dtype=np.int64
        )
        if np.any(self.parent_idx >= np.arange(len(self.nodes))):
            raise ValueError("Index rows must be in depth-first order (parents before children)")
        self.children: list[list[int]] = [[] for _ in self.nodes]
        for i, p in enumerate(self.parent_idx):
            if p >= 0:
                self.children[p].append(i)

    def __len__(self) -> int:
        return len(self.nodes)

    def hashes(self) -> dict[str, str]:
        return dict(zip(self.symbols, self.hashes_list, strict=True))

    def node(self, symbol: str) -> IpcNode:
        return self.scheme.node(symbol)

    def text_of(self, symbol: str) -> str:
        return self.texts[self.position[symbol]]

    # -- persistence ---------------------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, object] = {
            "symbol": pa.array(self.symbols),
            "path": pa.array([SYMBOL_SEPARATOR.join(p) for p in self.scheme.paths]),
            "text_hash": pa.array(self.hashes_list),
        }
        for j, name in enumerate(vector_columns(self.meta.dim)):
            columns[name] = pa.array(self.vectors[:, j], type=pa.float32())
        table = pa.table(columns).replace_schema_metadata(
            {META_KEY: json.dumps(asdict(self.meta)).encode()}
        )
        pq.write_table(table, path, compression="zstd")

    @staticmethod
    def read_meta(path: Path) -> IndexMeta:
        raw = pq.read_schema(path).metadata or {}
        if META_KEY not in raw:
            raise ValueError(f"{path} carries no text2ipc metadata")
        return IndexMeta(**json.loads(raw[META_KEY]))

    @classmethod
    def read(cls, path: Path, scheme: SchemeTable | Path) -> IpcIndex:
        meta = cls.read_meta(path)
        if isinstance(scheme, Path):
            scheme = SchemeTable.read(scheme)
        table = pq.read_table(path)
        symbols = table.column("symbol").to_pylist()
        if symbols != [n.symbol for n in scheme.nodes]:
            raise ValueError(f"{path.name} rows do not match the scheme table")
        cols = vector_columns(meta.dim)
        vectors = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in cols]
        ).astype(np.float32, copy=False)
        return cls(meta, scheme, vectors, table.column("text_hash").to_pylist())
