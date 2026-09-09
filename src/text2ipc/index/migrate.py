"""One-off conversion of the legacy CSV indexes (ADR 0002) to parquet (ADR 0006).

The CSV rows carry the rendered text and its hash, so the conversion can verify that
the new scheme table renders exactly the text the vectors were computed from, and no
embedding is needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import OVERLAY_LANGS
from ..scheme import SchemeTable, apply_titles, parse_scheme
from ..scheme.download import scheme_path
from ..scheme.inpi import fetch_inpi_titles
from .paths import index_path, scheme_table_path
from .store import IndexMeta, IpcIndex


def migrate_csv_index(csv_file: Path) -> tuple[Path, Path, str]:
    root = csv_file.resolve().parents[1]
    meta_raw = json.loads(csv_file.with_suffix(".meta.json").read_text())
    meta_raw.pop("style", None)
    meta = IndexMeta(**meta_raw)

    xml = scheme_path(meta.version, "EN" if meta.lang in OVERLAY_LANGS else meta.lang, root)
    nodes = parse_scheme(xml)
    if meta.lang in OVERLAY_LANGS:
        nodes = apply_titles(nodes, fetch_inpi_titles(meta.version, root=root))
    scheme = SchemeTable.from_nodes(nodes)
    hashes = scheme.hashes()

    vectors = np.zeros((len(scheme), meta.dim), dtype=np.float32)
    seen = 0
    mismatched: list[str] = []
    with open(csv_file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            i = scheme.position[row["symbol"]]
            if row["text_hash"] != hashes[i]:
                mismatched.append(row["symbol"])
            vectors[i] = np.fromstring(row["embedding"], sep=" ", dtype=np.float32)
            seen += 1
    if seen != len(scheme):
        raise ValueError(f"{csv_file.name}: {seen} rows for {len(scheme)} scheme entries")
    if mismatched:
        raise ValueError(
            f"{csv_file.name}: rendered text differs for {len(mismatched)} entries "
            f"(e.g. {mismatched[:3]}); rebuild instead of migrating"
        )

    scheme_target = scheme_table_path(meta.version, meta.lang, root)
    scheme.write(scheme_target)
    index = IpcIndex(meta, scheme, vectors, hashes)
    index_target = index_path(meta.version, meta.lang, meta.model, root)
    index.write(index_target)
    csv_mb = csv_file.stat().st_size / 1e6
    pq_mb = index_target.stat().st_size / 1e6
    report = (
        f"{meta.version} {meta.lang} {meta.model}: {seen} rows, all texts identical; "
        f"{csv_mb:.0f} MB CSV -> {pq_mb:.0f} MB parquet"
    )
    return scheme_target, index_target, report
