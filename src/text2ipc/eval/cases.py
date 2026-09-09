"""Eval cases: a text and the IPC symbols a patent office assigned to it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str  # what gets classified: abstract when available, else the title
    ipc: tuple[str, ...]  # canonical 14-char gold symbols
    lang: str = "pt"
    title: str | None = None
    abstract: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def save_cases(cases: list[EvalCase], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_many(
    paths: list[Path | str] | str,
    *,
    sample: int | None = None,
    seed: int = 0,
    stratify_by_abstract: bool = True,
) -> list[EvalCase]:
    """Load several JSONL files (or a glob) and optionally draw a random sample.

    With ``stratify_by_abstract`` the sample takes half its cases from those that have
    an abstract and half from title-only ones, as far as each pool allows.
    """
    import glob
    import random

    files = sorted(glob.glob(paths)) if isinstance(paths, str) else list(paths)
    cases = [c for f in files for c in load_cases(f)]
    if sample is None or sample >= len(cases):
        return cases
    rng = random.Random(seed)
    if not stratify_by_abstract:
        return rng.sample(cases, sample)
    with_abs = [c for c in cases if c.abstract]
    without = [c for c in cases if not c.abstract]
    n_abs = min(len(with_abs), sample // 2)
    n_wo = min(len(without), sample - n_abs)
    n_abs = min(len(with_abs), sample - n_wo)
    picked = rng.sample(with_abs, n_abs) + rng.sample(without, n_wo)
    rng.shuffle(picked)
    return picked


def load_cases(path: Path | str) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                d["ipc"] = tuple(d["ipc"])
                d["tags"] = tuple(d.get("tags", ()))
                cases.append(EvalCase(**d))
    return cases
