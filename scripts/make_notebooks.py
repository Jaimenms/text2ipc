"""Generate the study notebooks from code, so they stay reproducible and thin.

uv run python scripts/make_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_text2ipc.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_rerank.ipynb
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
`group` or `subgroup`. Results are ranked by cosine similarity between the text and
each entry's path vector; support from the ancestors and the subtree exist as
`Weights` but are off by default, since on the Portuguese scheme they lowered every
hit rate (`docs/evals.md`, ADR 0008). The `path` column is the whole chain of titles
from the section down to the entry, exactly the text that was embedded, so a result
reads as its own explanation. After ranking, results that sit on the same branch as
a better-ranked one (an ancestor or a descendant of it) are dropped, so each row is
a distinct branch of the IPC tree. A text with several paragraphs (a title above an
abstract) is embedded paragraph by paragraph and the vectors are averaged; a
paragraph longer than the model's 512 tokens is cut into sentence chunks.
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
  IPC entries, abstracts are prose. Averaging the title vector and the abstract
  vector adds 4.6 points at subclass@1 over the abstract alone, while concatenating
  the two strings adds nothing (`docs/evals.md`); that is why the classifier embeds
  paragraphs separately.
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
## 8. What comes next

1. The second stage: a cross-encoder re-judges the top candidates and lifts
   subclass@1 by 8 points (`02_rerank.ipynb`).
2. A stronger multilingual embedder (`bge-m3`, `multilingual-e5-large`): the first
   stage, not the heuristics, is the bottleneck (`docs/evals.md`).
3. kNN over similar published applications (vote on their IPC symbols) as a second
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


def notebook_02():
    cells = [
        md("""
# 02 · The second stage: a cross-encoder re-judges the candidates

The first stage of `text2ipc` embeds the text and the 80k IPC entries separately and
ranks by cosine. It finds the right entries far more often than it puts them first:
with 50 candidates, the office's subclass is in the list 64% of the time but first
only 24% of the time (`docs/evals.md`). This notebook shows the second stage that
closes part of that gap: a **cross-encoder** reads the text and an entry's path
together and gives a relevance verdict, and the two scores are fused.

Prerequisites: the Portuguese index from notebook 01, plus the reranker model, which
`sentence-transformers` downloads on first use (`BAAI/bge-reranker-v2-m3`, 2.2 GB).
On an Apple M-series GPU it judges about 70 pairs per second.
"""),
        code("""
import time
from pathlib import Path

import pandas as pd

from text2ipc import IpcClassifier
from text2ipc.eval import fusion_results, judge_candidates, load_many

ROOT = Path.cwd() if (Path.cwd() / "evals").is_dir() else Path.cwd().parent
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)


def as_frame(matches):
    return pd.DataFrame(
        [
            {
                "symbol": m.pretty,
                "score": round(m.score, 3),
                "sim": round(m.similarity, 3),
                "judge": None if m.judge is None else round(m.judge, 2),
                "entry": m.title,
            }
            for m in matches
        ]
    )


pt = IpcClassifier("20260101", lang="PT")
print(pt.version, pt.lang, pt.index.meta.model)
"""),
        md("""
## 1. One application, before and after

The abstract of a real application from RPI 2131 (PI 0620431-7), filed by INPI
under A61K 9/70: medicinal preparations in patches for dermal or transdermal
delivery. It also talks about hair care, which pulls the first stage towards
hairdressing (A45D). The eval text is the abstract alone, as here.
"""),
        code("""
text = (
    "A presente invenção refere-se ao uso de poliuretanos formadores de película que "
    "encontram utilização em agentes para o cuidado dos cabelos ou de misturas desses "
    "poliuretanos com outros polimeros em preparações farmacêuticas para a aplicação "
    "dérmica ou transdérmica de substâncias ativas, bem como emplastros e preparações "
    "farmacêuticas contendo esses poliuretanos para o cuidado dos cabelos."
)
first = pt.classify(text, level="subgroup", top_k=10)
as_frame(first)
"""),
        code("""
t0 = time.time()
reranked = pt.classify(text, level="subgroup", top_k=10, rerank=True)
print(f"reranked in {time.time() - t0:.1f} s (the first call also loads the model)")
as_frame(reranked)
"""),
        md("""
`score` is now `cosine × (0.5 + 0.5 × judge)`, where `judge` is the cross-encoder's
verdict in 0..1: an entry the judge dislikes keeps at most half its cosine. Scores
across the two tables are therefore on different scales; within one table the order
is what matters. `rerank=True` judged 50 candidates (`max(5 × top_k, 25)`) and
returned the best 10: A61K 47/60 (pharmaceutical carriers) moves to the top, the
hairdressing entries slide down. The verdicts are small (0.08 against 0.05), which
is the next point.

One caveat found while writing this: with the application's title added as a
second paragraph (a long line in capitals), the first stage drifts to other entries
and the judge gives almost every candidate a verdict near 0.99. The cross-encoder is
sensitive to the form of the text; the evals, and this notebook, use the abstract.
"""),
        md("""
## 2. The judge's verdicts

`clf.reranker()` gives the model itself. Its logits are not calibrated: the
offset changes from text to text, so a verdict of 0.1 or 0.9 says little on its
own and unrelated entries can still get a positive logit. What carries the signal
is the order within one query, and the `blend` fusion lets the cosine break the
near-ties the judge leaves.
"""),
        code("""
judge = pt.reranker()
pairs = [(m.pretty, m.title) for m in first[:6]]
logits = judge.score(text, [pt.index.text_of(m.symbol) for m in first[:6]])
pd.DataFrame(
    {"symbol": [p[0] for p in pairs], "logit": logits.round(2), "entry": [p[1] for p in pairs]}
)
"""),
        md("""
## 3. Evaluation: one judged run, several fusions

Judging is the expensive part, so `judge_candidates` runs the first stage and the
reranker once per case and keeps the candidates with both scores; `fusion_results`
then evaluates every fusion rule from that one run. 200 cases (100 with abstracts,
100 title-only), up to 25 candidates each (fewer where branches collapse): about
4,000 pairs.
"""),
        code("""
cases = load_many(str(ROOT / "evals" / "rpi_*.jsonl"), sample=200, seed=0)
t0 = time.time()
judged = judge_candidates(pt, cases, candidates=25, level="subgroup")
elapsed = time.time() - t0
pairs = sum(len(j.matches) for j in judged)
rate = pairs / elapsed
print(f"{len(judged)} cases, {pairs} pairs judged in {elapsed:.0f} s ({rate:.0f} pairs/s)")
"""),
        code("""
results = fusion_results(judged, top_k=10, level="subgroup")
rows = []
for name, result in results.items():
    for level in ("subclass", "group", "subgroup"):
        rows.append(
            {
                "fusion": name,
                "level": level,
                "hit@1": result.rate(level, 1),
                "hit@3": result.rate(level, 3),
                "hit@10": result.rate(level, 10),
            }
        )
table = pd.DataFrame(rows).pivot(index="level", columns="fusion", values="hit@1")
table = table.loc[["subclass", "group", "subgroup"], ["first", "judge", "product", "blend"]]
table.apply(lambda col: col.map("{:.1%}".format))
"""),
        md("""
### Reading the table

- `first` is the first stage alone; `judge` orders by the cross-encoder alone;
  `product` is `cosine × judge`; `blend`, the default, is `cosine × (0.5 + 0.5 × judge)`.
- On the full 991-case run (`docs/evals.md`) `blend` takes subclass@1 from 23.6% to
  31.6% and group@1 from 11.2% to 16.6%; 200 cases reproduce the direction with
  wider error bars (about ±3 points at this size).
- The judge decides and the cosine breaks ties: `judge` and `blend` land close, and
  both beat using the judge only as a tie-break (`docs/evals.md`).
"""),
        code("""
pd.DataFrame(rows).pivot(index="level", columns="fusion", values="hit@10").loc[
    ["subclass", "group", "subgroup"], ["first", "judge", "product", "blend"]
].apply(lambda col: col.map("{:.1%}".format))
"""),
        md("""
## 4. Where the reranker helps, and where it cannot

A case the second stage fixes: the right subclass was in the candidates but not
first. A case it cannot fix: the right entry was not among the candidates at all,
which no reordering can repair. Both are visible by comparing the first-stage order
with the fused order.
"""),
        code("""
from text2ipc.eval import fused_order
from text2ipc.eval.harness import truncate

fixed, lost = [], []
for j in judged:
    gold = {truncate(s, "subclass") for s in j.case.ipc}
    before = [truncate(m.symbol, "subclass") for m in j.matches[:10]]
    after = [truncate(m.symbol, "subclass") for m in fused_order(j, "blend", 10)]
    if before[0] not in gold and after[0] in gold:
        fixed.append(j)
    if not any(s in gold for s in (truncate(m.symbol, "subclass") for m in j.matches)):
        lost.append(j)
print(len(fixed), "cases fixed at subclass@1;", len(lost), "cases with no gold subclass among 25")
j = fixed[0]
print(j.case.id, "gold", j.case.ipc[:2], "|", j.case.text[:110])
pd.DataFrame(
    {
        "first stage": [m.pretty for m in j.matches[:5]],
        "reranked": [m.pretty for m in fused_order(j, "blend", 5)],
        "judge": [round(m.judge, 2) for m in fused_order(j, "blend", 5)],
    }
)
"""),
        md("""
## 5. Cost and where it runs

- Package: `t2ipc classify --rerank`, `t2ipc eval --rerank`, or `rerank=True`. The
  model is downloaded once; the cross-encoder adds well under a second per query on
  a GPU and a few seconds on CPU.
- Endpoint: `{"parameters": {"rerank": true}}` on the Hugging Face handler.
- Browser: the Space runs `jina-reranker-v2-base-multilingual` in 8 bits inside a
  Web Worker, about one second per candidate; on 300 cases it equals the bge model
  at @1 (`docs/evals.md`).
- Ceiling: with 50 candidates the subclass is present 64% of the time; the reranker
  reaches 32% at rank 1. The next gains are in the first stage (a stronger embedder)
  or in a reranker trained on patent data.
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
    for name, build in (("01_text2ipc.ipynb", notebook_01), ("02_rerank.ipynb", notebook_02)):
        nbf.write(build(), NB_DIR / name)
        print("wrote", NB_DIR / name)
