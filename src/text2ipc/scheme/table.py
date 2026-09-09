"""The scheme as a table: one row per entry, with the symbol path to the root.

This is what gets embedded and what is shown to users. The text of an entry is the
concatenation, top-down, of the titles along its path (section > class > subclass >
main group > subgroup > ... > the entry), with the guidance heading inserted before a
main group that sits under one. A subgroup's vector is the embedding of that whole
text, never of its own title alone.

Stored as ``scheme/ipc_<version>_<lang>.parquet`` with columns
``symbol, level, depth, parent, title, heading, path`` where ``path`` is the
``|``-joined chain of canonical symbols from the section to the entry.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .model import PATH_SEPARATOR, SYMBOL_SEPARATOR, IpcNode, text_hash

COLUMNS = ("symbol", "level", "depth", "parent", "title", "heading", "path")


@dataclass
class SchemeTable:
    nodes: list[IpcNode]
    paths: list[tuple[str, ...]]  # symbols from section to self, aligned with nodes

    def __post_init__(self):
        self.position = {n.symbol: i for i, n in enumerate(self.nodes)}

    @classmethod
    def from_nodes(cls, nodes: list[IpcNode]) -> SchemeTable:
        paths: dict[str, tuple[str, ...]] = {}
        out = []
        for n in nodes:  # parents precede children in scheme order
            if n.parent is None:
                path: tuple[str, ...] = (n.symbol,)
            else:
                path = (*paths[n.parent], n.symbol)
            paths[n.symbol] = path
            out.append(path)
        return cls(nodes, out)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[IpcNode]:
        return iter(self.nodes)

    def node(self, symbol: str) -> IpcNode:
        return self.nodes[self.position[symbol]]

    def path_of(self, symbol: str) -> tuple[str, ...]:
        return self.paths[self.position[symbol]]

    def text_of(self, symbol: str) -> str:
        return self.text_at(self.position[symbol])

    def text_at(self, i: int) -> str:
        parts = []
        for s in self.paths[i]:
            n = self.nodes[self.position[s]]
            if n.heading:
                parts.append(n.heading)
            parts.append(n.title)
        return PATH_SEPARATOR.join(p for p in parts if p)

    def texts(self) -> list[str]:
        return [self.text_at(i) for i in range(len(self.nodes))]

    def hashes(self) -> list[str]:
        return [text_hash(t) for t in self.texts()]

    # -- persistence ---------------------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "symbol": [n.symbol for n in self.nodes],
                "level": [n.level for n in self.nodes],
                "depth": pa.array([n.depth for n in self.nodes], type=pa.int8()),
                "parent": [n.parent for n in self.nodes],
                "title": [n.title for n in self.nodes],
                "heading": [n.heading for n in self.nodes],
                "path": [SYMBOL_SEPARATOR.join(p) for p in self.paths],
            }
        )
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def read(cls, path: Path) -> SchemeTable:
        t = pq.read_table(path, columns=list(COLUMNS)).to_pydict()
        nodes = [
            IpcNode(
                symbol=t["symbol"][i],
                level=t["level"][i],
                depth=int(t["depth"][i]),
                parent=t["parent"][i] or None,
                title=t["title"][i],
                heading=t["heading"][i] or "",
            )
            for i in range(len(t["symbol"]))
        ]
        paths = [tuple(p.split(SYMBOL_SEPARATOR)) for p in t["path"]]
        return cls(nodes, paths)
