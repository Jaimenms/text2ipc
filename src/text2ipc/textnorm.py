"""Small text normalisations shared by scheme texts and queries."""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")


def paragraphs(text: str) -> list[str]:
    """Non-empty parts separated by a blank line; single newlines do not split."""
    return [p for p in _PARAGRAPH.split(text) if p.strip()]


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace inside each paragraph; keep blank lines as breaks."""
    return "\n\n".join(" ".join(p.split()) for p in paragraphs(text))


def lowercase_if_shouting(text: str, threshold: float = 0.6) -> str:
    """Lower-case text whose letters are mostly upper case; leave mixed case alone.

    IPC section, class and subclass titles and RPI titles are printed in capitals,
    which multilingual tokenizers split badly. "DNA" inside a sentence-case title
    stays as is.
    """
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > threshold:
        return text.lower()
    return text
