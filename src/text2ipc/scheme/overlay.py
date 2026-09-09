"""Replace entry titles with a translation supplied as ``symbol,title`` rows.

WIPO master files exist in English and French. Other languages (Portuguese via INPI
Brazil, for example) come as title tables. Entries missing from the table keep their
original title, so partial translations degrade gracefully. Guidance headings have no
symbol and stay in the source language.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from .model import IpcNode
from .symbols import normalize_symbol


def load_titles_csv(path: Path | str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or {"symbol", "title"} - set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns 'symbol' and 'title'")
        for row in reader:
            title = (row["title"] or "").strip()
            if title:
                mapping[normalize_symbol(row["symbol"])] = " ".join(title.split())
    return mapping


def apply_titles(nodes: list[IpcNode], titles: dict[str, str]) -> list[IpcNode]:
    titles = {normalize_symbol(k): v for k, v in titles.items()}
    return [replace(n, title=titles.get(n.symbol, n.title)) for n in nodes]
