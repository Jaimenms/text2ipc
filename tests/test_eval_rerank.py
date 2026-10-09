"""One judged run, several fusion rules, on the mini scheme with the hash reranker."""

from __future__ import annotations

from text2ipc.classifier import IpcClassifier
from text2ipc.eval import EvalCase, fused_order, fusion_results, judge_candidates


def test_judge_once_and_fuse_many(mini_home):
    clf = IpcClassifier("20260101", lang="EN", model="hash:64", root=mini_home)
    cases = [
        EvalCase(id="a", text="hand tools spades shovels with teeth", ipc=("A01B0001040000",)),
        EvalCase(id="b", text="protocols network nodes", ipc=("H04L0067000000",)),
    ]
    seen = []
    judged = judge_candidates(
        clf, cases, candidates=5, reranker="hash:64", progress=lambda i, n: seen.append((i, n))
    )
    assert seen == [(1, 2), (2, 2)]
    assert all(len(j.logits) == len(j.matches) <= 5 for j in judged)
    results = fusion_results(judged, top_k=3, level="group")
    assert set(results) == {"first", "judge", "blend", "product"}
    assert all(r.n == 2 for r in results.values())
    assert results["first"].rate("subclass", 1) == 1.0
    reordered = fused_order(judged[0], "blend", 3)
    assert reordered[0].judge is not None and 0.0 <= reordered[0].judge <= 1.0
    assert [m.score for m in reordered] == sorted((m.score for m in reordered), reverse=True)
    assert fused_order(judged[0], "first", 2) == judged[0].matches[:2]
