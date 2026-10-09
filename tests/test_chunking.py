"""Long texts: sentence chunks that fit the embedder, and how chunk vectors combine."""

from __future__ import annotations

import numpy as np
import pytest

from text2ipc.chunking import split_text
from text2ipc.classifier import IpcClassifier
from text2ipc.embeddings.hashing import HashEmbedder
from text2ipc.search import SearchParams, search


def count(text: str) -> int:
    return len(text.split())


def test_split_packs_sentences_and_overlaps():
    text = "One two three. Four five six! Seven eight.\n\nNine ten eleven twelve; thirteen."
    chunks = split_text(text, 7, count, overlap=1)
    assert all(count(c) <= 7 for c in chunks)
    assert chunks[0] == "One two three. Four five six!"
    assert chunks[1].startswith("Four five six!")  # the last sentence is repeated
    assert "thirteen." in chunks[-1]
    assert split_text("", 7, count) == []


def test_split_cuts_a_sentence_longer_than_the_limit_by_words():
    text = " ".join(f"w{i}" for i in range(23))  # no sentence end at all
    chunks = split_text(text, 10, count, overlap=0)
    assert [count(c) for c in chunks] == [10, 10, 3]
    assert " ".join(chunks).split() == text.split()


def test_paragraphs_are_never_merged():
    assert split_text("A b.\n\nC d.", 10, count) == ["A b.", "C d."]
    assert split_text("TÍTULO\n\n\nresumo  do  pedido", None, count) == [
        "TÍTULO",
        "resumo do pedido",
    ]


def test_split_without_overlap_keeps_every_sentence_once():
    text = "A b c. D e f. G h i. J k l."
    chunks = split_text(text, 6, count, overlap=0)
    assert chunks == ["A b c. D e f.", "G h i. J k l."]


class LimitedHash(HashEmbedder):
    """The hash embedder with a pretend token limit, to exercise chunking offline."""

    @property
    def max_tokens(self) -> int | None:
        return 6


@pytest.fixture
def limited_clf(mini_home):
    clf = IpcClassifier("20260101", lang="EN", model=LimitedHash(64), root=mini_home)
    return clf


LONG = (
    "Hand tools for soil working. Spades and shovels with teeth. "
    "Hoes and hand cultivators with two or more blades. Harrows with non-rotating tools."
)


def test_short_text_is_embedded_whole(limited_clf):
    limited_clf.classify("spades shovels")
    assert limited_clf.last_chunks == 1


def test_title_and_abstract_paragraphs_are_averaged(limited_clf):
    limited_clf.classify("SPADES\n\nShovels with teeth.")
    assert limited_clf.last_chunks == 2
    limited_clf.classify("SPADES\n\nShovels with teeth.", chunking="truncate")
    assert limited_clf.last_chunks == 1


def test_long_text_is_chunked_and_mean_vector_searched(limited_clf):
    hits = limited_clf.classify(LONG, level="group")
    assert limited_clf.last_chunks >= 3
    assert hits and hits[0].symbol.startswith("A01B")
    truncated = limited_clf.classify(LONG, level="group", chunking="truncate")
    assert limited_clf.last_chunks == 1
    assert truncated


def test_max_chunking_scores_an_entry_by_its_best_chunk(limited_clf, mini_index):
    stack = limited_clf.embed_text(LONG, chunking="max")
    assert stack.ndim == 2 and stack.shape[0] == limited_clf.last_chunks
    hits = search(mini_index, stack, SearchParams(level="subgroup", top_k=3))
    best = max(
        search(mini_index, v, SearchParams(level="subgroup", top_k=1))[0].similarity for v in stack
    )
    assert abs(hits[0].similarity - best) < 1e-6
    with pytest.raises(ValueError):
        limited_clf.embed_text(LONG, chunking="median")


def test_hash_embedder_counts_tokens_and_has_no_limit():
    e = HashEmbedder(16)
    assert e.max_tokens is None and e.count_tokens("Spade with teeth, 2 blades") == 5
    assert e.embed_queries(["a", "b"]).shape == (2, 16)
    assert np.allclose(np.linalg.norm(e.embed_queries(["a b"]), axis=1), 1.0)
