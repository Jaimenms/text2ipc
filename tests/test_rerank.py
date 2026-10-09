"""Second stage: a reranker judges the first stage's candidates and the scores fuse."""

from __future__ import annotations

import numpy as np
import pytest

from text2ipc.classifier import IpcClassifier
from text2ipc.rerank import fuse, get_reranker, sigmoid
from text2ipc.rerank.hashing import HashReranker


def test_fusion_rules():
    scores = np.array([0.9, 0.8])
    logits = np.array([-20.0, 20.0])
    blend = fuse(scores, logits, "blend")
    assert blend[0] == pytest.approx(0.45, abs=1e-6) and blend[1] == pytest.approx(0.8, abs=1e-6)
    assert fuse(scores, logits, "judge")[1] == pytest.approx(1.0, abs=1e-6)
    assert fuse(scores, logits, "product")[0] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        fuse(scores, logits, "sum")
    assert sigmoid(np.array([0.0]))[0] == 0.5


def test_get_reranker_specs():
    assert isinstance(get_reranker("hash:32"), HashReranker)
    assert get_reranker("hash:32").name == "hash:32"
    r = HashReranker(16)
    assert get_reranker(r) is r
    with pytest.raises(ValueError):
        get_reranker("nope")
    with pytest.raises(ValueError):
        get_reranker("bogus:model")
    assert r.score("x", []).shape == (0,)


@pytest.fixture
def clf(mini_home):
    return IpcClassifier("20260101", lang="EN", model="hash:64", root=mini_home)


def test_rerank_reorders_by_fused_score_and_reports_the_verdict(clf):
    text = "hoes hand cultivators with two or more blades"
    plain = clf.classify(text, level="subgroup", top_k=3)
    reranked = clf.classify(text, level="subgroup", top_k=3, rerank="hash:64")
    assert all(m.judge is None for m in plain)
    assert all(m.judge is not None and 0.0 <= m.judge <= 1.0 for m in reranked)
    assert [m.score for m in reranked] == sorted((m.score for m in reranked), reverse=True)
    assert len(reranked) <= 3
    assert reranked[0].symbol == "A01B0001100000"  # the best bag-of-words match stays first


def test_rerank_judges_more_candidates_than_it_returns(clf):
    text = "harrows"
    few = clf.classify(text, level="group", top_k=1, rerank="hash:64", candidates=1)
    many = clf.classify(text, level="group", top_k=1, rerank="hash:64", candidates=10)
    assert len(few) == 1 and len(many) == 1
    judged_only = clf.classify(text, level="group", top_k=2, rerank="hash:64", fusion="judge")
    assert [m.score for m in judged_only] == [m.judge for m in judged_only]
    with pytest.raises(ValueError):
        clf.classify(text, rerank="hash:64", fusion="sum")


def test_reranker_is_loaded_once(clf):
    a = clf.reranker("hash:64")
    assert clf.reranker("hash:64") is a and a.name == "hash:64"
