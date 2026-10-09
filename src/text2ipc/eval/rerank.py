"""Measure the second stage without paying for it more than once.

``judge_candidates`` runs the first stage and the reranker over a list of cases and
keeps the candidates with their first-stage scores and judge logits;
``fusion_results`` then evaluates several ways of combining the two scores from that
single run (notebook 02 and the tables in docs/evals.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..rerank import Reranker, fuse, sigmoid
from ..search import Match, SearchParams, search
from .cases import EvalCase
from .harness import EvalResult, evaluate

FUSIONS: tuple[str, ...] = ("first", "judge", "blend", "product")


@dataclass(frozen=True)
class JudgedCase:
    case: EvalCase
    matches: list[Match]  # first-stage candidates, in first-stage order
    logits: np.ndarray  # one reranker logit per candidate


def judge_candidates(
    clf,
    cases: Sequence[EvalCase],
    *,
    candidates: int = 25,
    level: str = "subgroup",
    reranker: bool | str | Reranker = True,
    progress: Callable[[int, int], None] | None = None,
) -> list[JudgedCase]:
    """First stage (``candidates`` entries) plus reranker logits for every case."""
    from ..classifier import normalize_query

    judge = clf.reranker(reranker)
    params = SearchParams(level=level, top_k=candidates)
    out = []
    for i, case in enumerate(cases, 1):
        text = normalize_query(case.text)
        matches = search(clf.index, clf.embed_text(text), params)
        logits = judge.score(text, [m.text for m in matches]) if matches else np.zeros(0)
        out.append(JudgedCase(case, matches, np.asarray(logits, dtype=np.float32)))
        if progress:
            progress(i, len(cases))
    return out


def fused_order(judged: JudgedCase, fusion: str, top_k: int) -> list[Match]:
    """The candidate list reordered by one fusion rule (``first`` keeps stage one)."""
    if fusion == "first":
        return judged.matches[:top_k]
    scores = np.array([m.score for m in judged.matches])
    fused = fuse(scores, judged.logits, fusion)
    verdict = sigmoid(judged.logits)
    order = np.argsort(-fused, kind="stable")[:top_k]
    return [
        replace(judged.matches[i], score=float(fused[i]), judge=float(verdict[i])) for i in order
    ]


def fusion_results(
    judged: Sequence[JudgedCase],
    *,
    fusions: Sequence[str] = FUSIONS,
    top_k: int = 10,
    level: str = "subgroup",
) -> dict[str, EvalResult]:
    """Hit rates per fusion rule, from one judged run."""
    by_text = {j.case.text: j for j in judged}
    cases = [j.case for j in judged]
    return {
        f: evaluate(
            cases, lambda t, f=f: fused_order(by_text[t], f, top_k), level=level, top_k=top_k
        )
        for f in fusions
    }
