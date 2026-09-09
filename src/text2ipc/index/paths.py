"""Where scheme tables and indexes live, and how to find the previous version."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import home
from ..embeddings.base import model_slug

_NAME_RE = re.compile(r"^ipc_(?P<version>\d{8})_(?P<lang>[a-z]{2})_(?P<model>.+)\.parquet$")


@dataclass(frozen=True)
class IndexRef:
    version: str
    lang: str
    model_slug: str
    path: Path


def index_dir(root: Path | None = None) -> Path:
    return (root or home()) / "index"


def scheme_dir(root: Path | None = None) -> Path:
    return (root or home()) / "scheme"


def scheme_table_path(version: str, lang: str, root: Path | None = None) -> Path:
    return scheme_dir(root) / f"ipc_{version}_{lang.lower()}.parquet"


def index_path(version: str, lang: str, model: str, root: Path | None = None) -> Path:
    return index_dir(root) / f"ipc_{version}_{lang.lower()}_{model_slug(model)}.parquet"


def available_indexes(root: Path | None = None) -> list[IndexRef]:
    d = index_dir(root)
    if not d.exists():
        return []
    return [
        IndexRef(m["version"], m["lang"].upper(), m["model"], p)
        for p in sorted(d.glob("ipc_*.parquet"))
        if (m := _NAME_RE.match(p.name))
    ]


def find_previous_index(
    version: str, lang: str, model: str, root: Path | None = None
) -> Path | None:
    """Newest built index for the same language and model with an older version."""
    slug = model_slug(model)
    older = [
        r
        for r in available_indexes(root)
        if r.lang == lang.upper() and r.model_slug == slug and r.version < version
    ]
    return older[-1].path if older else None
