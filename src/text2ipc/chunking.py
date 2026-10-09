"""Split a long text into pieces an embedder can take whole.

Embedders have a token limit (512 for the e5 family) beyond which they truncate
silently, so a full patent description would be judged by its first page. Instead
the text is cut at paragraph and sentence ends into chunks that fit, each chunk is
embedded as a query, and the chunk vectors are combined (``IpcClassifier.embed_text``).
"""

from __future__ import annotations

import re
from collections.abc import Callable

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=\S)")


def split_text(
    text: str,
    max_tokens: int | None,
    count_tokens: Callable[[str], int],
    *,
    overlap: int = 1,
) -> list[str]:
    """One chunk per paragraph, cut further into sentence chunks of at most
    ``max_tokens`` tokens when a paragraph is too long.

    Paragraphs are separated by a blank line and never merged: a title above an
    abstract gives two chunks. Within a paragraph, cuts happen at sentence ends,
    except for a single sentence over the limit, which is cut by words. ``overlap``
    repeats that many trailing sentences of a chunk at the start of the next.
    ``count_tokens`` must count a text the way the embedder will see it (prefix and
    special tokens included); ``max_tokens=None`` means no limit.
    """
    out: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        if paragraph.strip():
            out.extend(_pack(paragraph.strip(), max_tokens, count_tokens, overlap))
    return out


def _pack(
    paragraph: str, max_tokens: int | None, count_tokens: Callable[[str], int], overlap: int
) -> list[str]:
    if max_tokens is None:
        return [" ".join(paragraph.split())]
    units = [s.strip() for s in _SENTENCE.split(paragraph) if s.strip()]
    units = [u for unit in units for u in _cut_by_words(unit, max_tokens, count_tokens)]
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and count_tokens(" ".join([*current, unit])) > max_tokens:
            chunks.append(current)
            carry = current[-overlap:] if overlap > 0 else []
            current = [*carry, unit]
            while len(current) > 1 and count_tokens(" ".join(current)) > max_tokens:
                current = current[1:]
        else:
            current.append(unit)
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def _cut_by_words(unit: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    if count_tokens(unit) <= max_tokens:
        return [unit]
    out: list[str] = []
    words = unit.split()
    piece: list[str] = []
    for w in words:
        if piece and count_tokens(" ".join([*piece, w])) > max_tokens:
            out.append(" ".join(piece))
            piece = []
        piece.append(w)
    if piece:
        out.append(" ".join(piece))
    return out
