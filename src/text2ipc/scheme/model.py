"""The unit of the IPC hierarchy as this package sees it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

PATH_SEPARATOR = " > "
SYMBOL_SEPARATOR = "|"


@dataclass(frozen=True)
class IpcNode:
    """One classifiable IPC entry (section, class, subclass, main group or subgroup)."""

    symbol: str  # canonical WIPO form, e.g. A01B0001020000 or A01B
    level: str  # one of config.LEVELS
    depth: int  # dot level for subgroups (1 = one dot), 0 otherwise
    parent: str | None  # canonical symbol of the parent entry, None for sections
    title: str  # this entry's own title, reference notes stripped
    heading: str = ""  # guidance heading the entry sits under (main groups only)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
