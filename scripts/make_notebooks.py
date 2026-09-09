"""Generate the study notebooks from code, so they stay reproducible and thin.

uv run python scripts/make_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_text2ipc.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook_01():
    cells = [
        md("""
# 01 · text2ipc: from a patent text to IPC symbols

This notebook shows the public interface of `text2ipc` and evaluates it on 1,000
applications published by INPI Brazil (the *Revista da Propriedade Industrial*).

Prerequisites, run once from the repository root:

```bash
uv sync --all-extras
uv run t2ipc build --version 20260101 --lang PT   # Portuguese scheme, default model
uv run t2ipc build --version 20260101 --lang EN   # English scheme, for comparison
```

Everything below reads the Parquet tables under `data/` (or `$TEXT2IPC_HOME`).
"""),
        code("""
from pathlib import Path

import pandas as pd

from text2ipc import IpcClassifier
from text2ipc.eval import evaluate, load_many

ROOT = Path.cwd() if (Path.cwd() / "evals").is_dir() else Path.cwd().parent
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", None)


def as_frame(matches):
    # One row per match; `path` is the full text the entry was embedded with
    # (section > class > subclass > group > subgroups), which is also the explanation.
    return pd.DataFrame(
        [
            {
                "symbol": m.pretty,
                "level": m.level,
                "score": round(m.score, 3),
                "sim": round(m.similarity, 3),
                "path": m.text,
            }
            for m in matches
        ]
    )
"""),
        md("""
## 1. One classifier, one index

An `IpcClassifier` is bound to an IPC version, a scheme language and an embedding
model. Loading is lazy: the index (about 80k vectors) and the model load on the first
call. `lang="PT"` selects the scheme whose titles come from INPI's Portuguese
translation, the best choice for Portuguese text.
"""),
        code("""
pt = IpcClassifier("20260101", lang="PT")
print(pt.version, pt.lang, pt.index.meta.model, len(pt.index), "entries")
"""),
        md("""
## 2. Classify a Portuguese abstract

`level` picks the hierarchy level of the answer: `section`, `class`, `subclass`,
`group` or `subgroup`. Results are ranked by a score that combines the entry's own
similarity with support from its ancestors and its subtree. The `path` column is the
whole chain of titles from the section down to the entry, exactly the text that was
embedded, so a result reads as its own explanation. After ranking, results that sit on
the same branch as a better-ranked one (an ancestor or a descendant of it) are
dropped, so each row is a distinct branch of the IPC tree.
"""),
        code("""
# Modelled on an application from RPI 2100 (PI 0318341-6), which INPI filed under A62C 15/00.
abstract = "Aparelho para combate a incêndios com mangueira flexível reforçada e bico regulável."
as_frame(pt.classify(abstract, level="subclass", top_k=5))
"""),
        code("""
as_frame(pt.classify(abstract, level="group", top_k=5))
"""),
        md("""
Compare with what INPI assigned, A62C 15/00 (fire-fighting apparatus with hoses).
The typical shape of a good answer today is a correct subclass near the top and the
assigned main group within the first few candidates; below group level the ranking
is noisy. Scores sit in a narrow band because e5 models compress cosine similarity,
so the order matters more than the absolute value.
"""),
        md("""
## 3. As deep as the evidence supports

`level="auto"` walks down from the best subclasses along the best-scoring branch and
answers with the node on that walk whose own similarity is highest. Short or generic
texts stop high; specific ones reach subgroups.
"""),
        code("""
for text in ["Aparelhos de combate a incêndios", abstract]:
    print(text[:60], "...")
    display(as_frame(pt.classify(text, level="auto", top_k=3)))
"""),
        md("""
## 4. Reading a result: the path an entry was embedded with

Every vector is the embedding of the entry's full path text, from the section down to
the entry itself. The scheme table keeps that path as a chain of symbols, so a result
can be explained.
"""),
        code("""
from text2ipc.scheme import format_symbol

symbol = "A62C0015000000"  # A62C 15/00
print(" | ".join(format_symbol(s) for s in pt.index.scheme.path_of(symbol)))
print(pt.index.text_of(symbol))
"""),
        md("""
## 5. Same question, English scheme

The embedder is multilingual, so a Portuguese query can be matched against the
English scheme too. It works, but the Portuguese scheme scores higher on the evals
below because nothing has to cross languages.
"""),
        code("""
en = IpcClassifier("20260101", lang="EN")
as_frame(en.classify(abstract, level="group", top_k=5))
"""),
        md("""
## 6. Multi-label answers with a score gap

`gap` keeps every result within that distance of the best score, so the number of
answers depends on how confident the ranking is. Cosine scores of e5 models live in a
narrow band, so gaps are small numbers.
"""),
        code("""
as_frame(pt.classify(abstract, level="subclass", top_k=10, gap=0.005))
"""),
        md("""
## 7. Evaluation on 1,000 INPI applications

`evals/` holds one JSONL per RPI issue (29 issues, about 12k applications). Each case
has the Portuguese title, the abstract when the issue printed one, and the IPC symbols
the office assigned. We draw 1,000 cases with a fixed seed, half with abstracts and
half title-only, and measure hit@k per hierarchy level: a hit when any gold symbol,
truncated to the level, appears among the top-k predictions.
"""),
        code("""
cases = load_many(str(ROOT / "evals" / "rpi_*.jsonl"), sample=1000, seed=0)
with_abstract = [c for c in cases if c.abstract]
title_only = [c for c in cases if not c.abstract]
print(len(cases), "cases:", len(with_abstract), "with abstract,", len(title_only), "title-only")
print("issues:", sorted({c.id.split(":")[0] for c in cases})[:8], "...")
"""),
        code("""
def run(clf, subset, name):
    result = evaluate(subset, lambda t: clf.classify(t, top_k=10), level="subgroup", top_k=10)
    frame = pd.DataFrame(result.rows())
    frame.insert(0, "run", name)
    return frame


frames = []
for label, subset in [("abstract", with_abstract), ("title-only", title_only)]:
    frames.append(run(pt, subset, f"PT scheme · {label}"))
    frames.append(run(en, subset, f"EN scheme · {label}"))
results = pd.concat(frames, ignore_index=True)
percent = [c for c in results.columns if c.startswith(("hit", "main"))]
shown = results.copy()
shown[percent] = shown[percent].apply(lambda col: col.map("{:.1%}".format))
shown["mrr"] = shown["mrr"].map("{:.3f}".format)
shown
"""),
        md("""
### Reading the table

- **Subclass** is the level a searcher usually needs first; **group** is what a
  classification officer assigns. Subgroups are noisy for a pure embedding approach.
- On titles the Portuguese scheme is clearly ahead (about 4 points at subclass@1 and
  at group@10). On abstracts the two schemes tie within the noise of 500 cases; the
  abstracts come from 2011 to 2015 issues, whose gold symbols predate several IPC
  revisions.
- Titles score higher than abstracts at rank 1: titles are keyword lists shaped like
  IPC entries, abstracts are prose. Concatenating title and abstract is an obvious
  next experiment.
- Gold symbols from 2011 to 2015 issues predate several IPC revisions; mapping them
  through WIPO's concordance list would make the abstract rows fairer.
"""),
        code("""
pivot = results.pivot(index="level", columns="run", values="hit@10").loc[
    ["section", "class", "subclass", "group", "subgroup"]
]
pivot.apply(lambda col: col.map("{:.1%}".format))
"""),
        md("""
## 8. What to try next

1. A stronger multilingual embedder (`bge-m3`, `multilingual-e5-large`): the model,
   not the heuristics, is the bottleneck (`docs/evals.md`).
2. An LLM rerank over the 20 to 30 surviving candidates, with the path text as
   evidence.
3. `title + abstract` as the query text.
4. kNN over similar published applications (vote on their IPC symbols) as a second
   signal.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return nb


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(notebook_01(), NB_DIR / "01_text2ipc.ipynb")
    print("wrote", NB_DIR / "01_text2ipc.ipynb")
