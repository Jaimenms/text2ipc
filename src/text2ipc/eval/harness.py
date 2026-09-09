"""Hit-rate at every hierarchy level.

A case counts as a hit at level L and cut-off k if any of the top-k predictions,
truncated to level L, equals any gold symbol truncated to level L. Gold sets are
multi-label and the first listed symbol is the office's main one, so ``main_*``
metrics score only against that symbol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import LEVELS
from ..search import Match
from .cases import EvalCase


def truncate(symbol: str, level: str) -> str:
    if level == "section":
        return symbol[:1]
    if level == "class":
        return symbol[:3]
    if level == "subclass":
        return symbol[:4]
    if level == "group":
        return symbol[:8] + "000000" if len(symbol) == 14 else symbol
    return symbol


@dataclass
class EvalResult:
    level: str
    top_k: int
    n: int = 0
    hits: dict[str, dict[int, int]] = field(default_factory=dict)  # level -> k -> hits
    main_hits: dict[str, dict[int, int]] = field(default_factory=dict)
    reciprocal_rank: dict[str, float] = field(default_factory=dict)
    misses: list[tuple[str, str, list[str]]] = field(default_factory=list)  # id, gold, predicted

    def rate(self, level: str, k: int, *, main: bool = False) -> float:
        table = self.main_hits if main else self.hits
        return table.get(level, {}).get(k, 0) / self.n if self.n else 0.0

    def mrr(self, level: str) -> float:
        return self.reciprocal_rank.get(level, 0.0) / self.n if self.n else 0.0

    def rows(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> list[dict]:
        """One dict per level, ready for ``pandas.DataFrame``."""
        ks = tuple(k for k in ks if k <= self.top_k)
        out = []
        for level in LEVELS[: LEVELS.index(self.level) + 1]:
            row = {"level": level, "n": self.n}
            row.update({f"hit@{k}": self.rate(level, k) for k in ks})
            row["main@1"] = self.rate(level, 1, main=True)
            row["mrr"] = self.mrr(level)
            out.append(row)
        return out

    def table(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> str:
        ks = tuple(k for k in ks if k <= self.top_k)
        head = "level      " + "".join(f"  hit@{k:<3}" for k in ks) + "  main@1   mrr"
        rows = [head]
        for level in LEVELS[: LEVELS.index(self.level) + 1]:
            cells = "".join(f"  {self.rate(level, k):6.1%}" for k in ks)
            rows.append(
                f"{level:<11}{cells}  {self.rate(level, 1, main=True):6.1%}  {self.mrr(level):.3f}"
            )
        return "\n".join(rows)


def evaluate(
    cases: list[EvalCase],
    classify: Callable[[str], list[Match]],
    *,
    level: str = "subgroup",
    top_k: int = 10,
    progress: Callable[[int, int], None] | None = None,
) -> EvalResult:
    result = EvalResult(level=level, top_k=top_k)
    levels = LEVELS[: LEVELS.index(level) + 1]
    for lv in levels:
        result.hits[lv] = dict.fromkeys(range(1, top_k + 1), 0)
        result.main_hits[lv] = dict.fromkeys(range(1, top_k + 1), 0)
        result.reciprocal_rank[lv] = 0.0

    for i, case in enumerate(cases, 1):
        preds = [m.symbol for m in classify(case.text)[:top_k]]
        result.n += 1
        for lv in levels:
            gold = {truncate(s, lv) for s in case.ipc}
            main = truncate(case.ipc[0], lv)
            seen: list[str] = []
            first_hit = None
            for p in preds:
                t = truncate(p, lv)
                if t in seen:
                    continue
                seen.append(t)
                if first_hit is None and t in gold:
                    first_hit = len(seen)
                    result.reciprocal_rank[lv] += 1.0 / first_hit
                if t == main:
                    for k in range(len(seen), top_k + 1):
                        result.main_hits[lv][k] += 1
            if first_hit is not None:
                for k in range(first_hit, top_k + 1):
                    result.hits[lv][k] += 1
            elif lv == level:
                result.misses.append((case.id, ",".join(sorted(gold)), seen[:5]))
        if progress:
            progress(i, len(cases))
    return result
