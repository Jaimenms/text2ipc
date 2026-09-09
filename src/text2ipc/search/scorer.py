"""Hierarchical scoring on top of flat cosine similarity.

Flat nearest-neighbour over 80k entries has two failure modes: deep entries with long
specific texts win on word overlap alone, and an isolated hit in an unrelated subtree
can outrank a well-supported one. Three heuristics address that:

1. **Path support** - the mean similarity of an entry's ancestors. A good subgroup
   usually sits under a good group, subclass and class.
2. **Subtree support** - the best similarity anywhere below an entry. A class whose
   subtree lights up is more plausible than one where only the class title matches.
3. **Beam descent** - candidates at a level are only considered under the top-``b``
   parents of the level above. A parent is ranked by the best composite score anywhere
   in its subtree, never by its own title (section and class titles are too generic to
   score), so the flat best candidate always survives pruning and the beam only removes
   candidates from weakly supported branches.

The final score is a weighted sum of own similarity, path support and subtree support.
``level="auto"`` walks down from the best subclasses along the best-scoring branch
and answers with the best node on that walk, which yields "as detailed as the
evidence supports".

**Branch de-duplication** runs last: the top-``k`` results are walked in rank order
and a result is dropped when it lies on the same branch as one already accepted,
i.e. it is an ancestor or a descendant of it. A subgroup nested inside the best
subgroup adds nothing a user could not read off the best one's path, so the list
keeps only distinct branches. The answer may therefore hold fewer than ``k`` rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from ..config import AUTO_LEVEL, LEVELS
from ..index.store import IpcIndex
from ..scheme.symbols import format_symbol


@dataclass(frozen=True)
class Weights:
    """Defaults come from the sweep in docs/evals.md; subtree support is kept for
    experiments but off by default since beam selection already uses it."""

    own: float = 0.7
    path: float = 0.3
    subtree: float = 0.0


@dataclass(frozen=True)
class Beam:
    """How many parents survive at each level before descending."""

    section: int = 5
    class_: int = 10
    subclass: int = 20
    group: int = 40

    def width(self, level: str) -> int:
        return getattr(self, "class_" if level == "class" else level)


@dataclass(frozen=True)
class SearchParams:
    level: str = "subgroup"
    top_k: int = 10
    weights: Weights = field(default_factory=Weights)
    beam: Beam = field(default_factory=Beam)
    #: Drop results scoring more than this below the best one (None keeps top_k).
    gap: float | None = None
    #: ``auto`` level: descend while best child >= parent - margin.
    auto_margin: float = 0.02
    #: ``auto`` level: how many subclasses to start descending from.
    auto_roots: int = 5
    #: Drop results that are ancestors or descendants of a higher-ranked result.
    dedupe_branches: bool = True


@dataclass(frozen=True)
class Match:
    symbol: str
    level: str
    depth: int
    title: str
    text: str
    score: float
    similarity: float
    path_support: float
    subtree_support: float

    @property
    def pretty(self) -> str:
        return format_symbol(self.symbol)


def search(index: IpcIndex, query: np.ndarray, params: SearchParams | None = None) -> list[Match]:
    params = params or SearchParams()
    if params.level != AUTO_LEVEL and params.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS} or {AUTO_LEVEL!r}")

    sims = index.vectors @ np.asarray(query, dtype=np.float32)
    path = _path_support(index, sims)
    subtree = _subtree_support(index, sims)
    w = params.weights
    score = w.own * sims + w.path * path + w.subtree * subtree

    best_below = _subtree_support(index, score)

    if params.level == AUTO_LEVEL:
        chosen = _auto_descend(index, sims, score, best_below, params)
    else:
        chosen = _beam_descend(index, best_below, params.level, params.beam)
        chosen = sorted(chosen, key=lambda i: -score[i])[: params.top_k]

    if params.gap is not None and chosen:
        best = score[chosen[0]]
        chosen = [i for i in chosen if score[i] >= best - params.gap]
    if params.dedupe_branches:
        chosen = _distinct_branches(index, chosen)

    return [
        Match(
            symbol=index.nodes[i].symbol,
            level=index.nodes[i].level,
            depth=index.nodes[i].depth,
            title=index.nodes[i].title,
            text=index.texts[i],
            score=float(score[i]),
            similarity=float(sims[i]),
            path_support=float(path[i]),
            subtree_support=float(subtree[i]),
        )
        for i in chosen
    ]


def _distinct_branches(index: IpcIndex, ranked: list[int]) -> list[int]:
    """Keep a result only if no accepted result sits on its path or below it."""
    accepted: list[int] = []
    accepted_chains: list[set[str]] = []
    for i in ranked:
        chain = set(index.scheme.paths[i])
        symbol = index.symbols[i]
        same_branch = any(
            index.symbols[a] in chain or symbol in a_chain
            for a, a_chain in zip(accepted, accepted_chains, strict=True)
        )
        if not same_branch:
            accepted.append(i)
            accepted_chains.append(chain)
    return accepted


def _path_support(index: IpcIndex, sims: np.ndarray) -> np.ndarray:
    """Mean similarity of the ancestors; equals own similarity for roots."""
    total = np.zeros_like(sims)
    count = np.zeros(len(sims), dtype=np.int32)
    for i, p in enumerate(index.parent_idx):  # parents precede children
        if p >= 0:
            total[i] = total[p] + sims[p]
            count[i] = count[p] + 1
    return np.where(count > 0, total / np.maximum(count, 1), sims)


def _subtree_support(index: IpcIndex, sims: np.ndarray) -> np.ndarray:
    """Max similarity over the node and everything below it."""
    best = sims.copy()
    parents = index.parent_idx
    for i in range(len(sims) - 1, -1, -1):  # children precede parents in reverse
        p = parents[i]
        if p >= 0 and best[i] > best[p]:
            best[p] = best[i]
    return best


def _beam_descend(index: IpcIndex, best_below: np.ndarray, target: str, beam: Beam) -> list[int]:
    """Walk down from the sections, keeping the parents whose subtree scores best."""
    target_rank = LEVELS.index(target)
    frontier = [i for i in range(len(index)) if index.levels[i] == 0]  # sections
    for rank, level in enumerate(LEVELS):
        if rank == target_rank:
            return frontier
        frontier = sorted(frontier, key=lambda i: -best_below[i])[: beam.width(level)]
        next_level = LEVELS[rank + 1]
        if next_level == "subgroup":
            frontier = list(_descendants(index, frontier))
        else:
            frontier = [c for i in frontier for c in index.children[i]]
    return frontier


def _descendants(index: IpcIndex, roots: Iterable[int]) -> Iterable[int]:
    stack = list(roots)
    while stack:
        i = stack.pop()
        for c in index.children[i]:
            yield c
            stack.append(c)


def _auto_descend(
    index: IpcIndex,
    sims: np.ndarray,
    score: np.ndarray,
    best_below: np.ndarray,
    params: SearchParams,
) -> list[int]:
    """From the best subclasses, walk down the branch with the best subtree score while
    that subtree still promises something within ``auto_margin`` of the current node,
    then answer with the node on the walked path whose *own* similarity is highest
    (deepest on ties). Own similarity is used because composite scores are not
    comparable across levels: deeper nodes inherit path support from their ancestors."""
    roots = _beam_descend(index, best_below, "subclass", params.beam)
    roots = sorted(roots, key=lambda i: -best_below[i])[: params.auto_roots]
    out: list[int] = []
    for i in roots:
        path = [i]
        node = i
        while index.children[node]:
            best_child = max(index.children[node], key=lambda c: best_below[c])
            if best_below[best_child] < score[node] - params.auto_margin:
                break
            node = best_child
            path.append(node)
        best = max(reversed(path), key=lambda j: sims[j])
        if best not in out:
            out.append(best)
    return sorted(out, key=lambda i: -score[i])[: params.top_k]
