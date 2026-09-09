"""Turn a WIPO ``ipc_scheme`` XML file into a flat, ordered list of :class:`IpcNode`.

Entries nest physically in the XML (subgroups inside their main group, groups inside
the subclass...). Alongside the classifiable entries there are:

- ``t``  subsection titles (plain headings between classes, no symbol of their own)
- ``g``  guidance headings that span a range of main groups
- ``n``  notes
- ``i``  catchword indexes

None of those are classification targets. Guidance headings are kept as context for
the entries that sit under them; the rest are skipped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from .model import IpcNode
from .symbols import level_of_kind

NS = "{http://www.wipo.int/classifications/ipc/masterfiles}"
TITLE_PART_SEPARATOR = "; "


def parse_scheme(path: Path | str) -> list[IpcNode]:
    """Entries in depth-first order, so every parent precedes its children."""
    root = ET.parse(path).getroot()
    return list(_walk(root, parent=None))


def _walk(element: ET.Element, *, parent: str | None) -> Iterator[IpcNode]:
    heading: tuple[str, str] | None = None  # (title, last symbol it covers)
    for child in element:
        if child.tag != f"{NS}ipcEntry":
            continue
        kind = child.get("kind", "")
        level = level_of_kind(kind)
        if level is None:
            if kind == "g":
                # A guidance heading is an empty leaf in the XML covering the sibling
                # main groups from ``symbol`` to ``endSymbol``.
                heading = (_title_of(child), child.get("endSymbol") or child.get("symbol", ""))
            continue
        symbol = child.get("symbol", "")
        under_heading = heading is not None and level == "group" and symbol <= heading[1]
        yield IpcNode(
            symbol=symbol,
            level=level,
            depth=int(kind) if kind.isdigit() else 0,
            parent=parent,
            title=_title_of(child),
            heading=heading[0] if under_heading and heading else "",
        )
        yield from _walk(child, parent=symbol)


def _title_of(entry: ET.Element) -> str:
    """Join the title parts, dropping cross references such as "(edge trimmers A01G 3/06)".

    Cross references point *away* from the entry; embedding them would attract exactly
    the texts that belong elsewhere.
    """
    body = entry.find(f"{NS}textBody/{NS}title")
    if body is None:
        return ""
    parts = []
    for part in body.findall(f"{NS}titlePart"):
        text = part.find(f"{NS}text")
        if text is not None:
            value = "".join(text.itertext()).strip()
            if value:
                parts.append(" ".join(value.split()))
    return TITLE_PART_SEPARATOR.join(parts)
