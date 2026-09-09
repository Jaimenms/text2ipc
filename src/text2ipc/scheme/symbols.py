"""IPC symbol formats.

WIPO master files use a fixed 14-character form: subclass (4) + main group (4, zero
padded) + subgroup (6, zero padded). ``A01B0001020000`` is ``A01B 1/02``. Sections,
classes and subclasses are 1, 3 and 4 characters respectively.
"""

from __future__ import annotations

import re

_CANON_RE = re.compile(r"^[A-H](\d{2}([A-Z](\d{4}\d{6})?)?)?$")
_PRETTY_RE = re.compile(r"^([A-H]\d{2}[A-Z])\s*(\d{1,4})/(\d{2,6})$")

_KIND_TO_LEVEL = {"s": "section", "c": "class", "u": "subclass", "m": "group"}


def level_of_kind(kind: str) -> str | None:
    """Map a WIPO ``kind`` attribute to a level, or None for non-classifiable kinds."""
    if kind in _KIND_TO_LEVEL:
        return _KIND_TO_LEVEL[kind]
    if kind.isdigit():
        return "subgroup"
    return None


def level_of_symbol(symbol: str) -> str:
    n = len(symbol)
    if n == 1:
        return "section"
    if n == 3:
        return "class"
    if n == 4:
        return "subclass"
    if n == 14:
        return "group" if symbol[8:] == "000000" else "subgroup"
    raise ValueError(f"Not a canonical IPC symbol: {symbol!r}")


def normalize_symbol(symbol: str) -> str:
    """Accept ``A01B 1/02``, ``A01B1/02`` or canonical form; return canonical form."""
    s = symbol.strip().upper()
    if _CANON_RE.match(s):
        return s
    if m := _PRETTY_RE.match(s):
        sub, group, subgroup = m.groups()
        return f"{sub}{int(group):04d}{subgroup.ljust(6, '0')}"
    raise ValueError(f"Unrecognised IPC symbol: {symbol!r}")


def format_symbol(symbol: str) -> str:
    """Canonical -> human form: ``A01B0001020000`` -> ``A01B 1/02``."""
    if len(symbol) != 14:
        return symbol
    sub, group, subgroup = symbol[:4], symbol[4:8], symbol[8:]
    subgroup = subgroup.rstrip("0")
    if len(subgroup) < 2:
        subgroup = subgroup.ljust(2, "0")
    return f"{sub} {int(group)}/{subgroup}"
