# Eval log

A log in chronological order: every run is kept, including the ones that were
rejected, so a section describes the setup of its day. The first sections used the
English scheme and e5-small; from "Text style and scheme language" on, the Portuguese
scheme; from "Path weight set to 0" on, the current defaults.

**State on 2026-10-09** (e5-base, PT scheme, path weight 0, paragraphs averaged):
subclass@1 23.2% and @10 38.4% on the 2100 abstracts, 29.4% and 45.7% on the 2905
titles; with the cross-encoder second stage, subclass@1 31.6% and group@1 16.6% on
the 1,000-case sample. The sections are: case files, query normalisation, beam and
weights, baselines, the 1,000-case sample, model comparison, text style and scheme
language, int8 vectors for the browser, auto level, path weight, candidate recall,
combining the parts of an application, cross-encoder reranking.

Cases of the first sections: `evals/rpi_2905.jsonl`, 578 Portuguese titles (no
abstracts) from RPI issue 2905 (2026-09-08), dispatches 1.3 and 3.1, gold = every IPC
symbol printed by INPI. Index: IPC 20260101, English scheme. Metric: hit@k if any
gold symbol truncated to the level appears in the top-k predictions truncated to
that level.

## Case files

29 RPI issues (2100 and 2905 chosen by hand, 27 sampled uniformly at random with seed
42 between them), one JSONL per issue under `evals/`. Abstracts (INID 57) appear in the
text-format issues up to about 2215 (2013), partially until 2329 (2015), and never
afterwards; from 2533 on the issues are XML with titles only.

| issue | cases | with abstract |
|---|---|---|
| 2100 | 557 | 557 |
| 2126 | 274 | 274 |
| 2128 | 374 | 374 |
| 2131 | 295 | 295 |
| 2133 | 267 | 267 |
| 2190 | 517 | 517 |
| 2196 | 115 | 115 |
| 2205 | 223 | 223 |
| 2215 | 375 | 375 |
| 2243 | 151 | 37 |
| 2304 | 213 | 45 |
| 2324 | 720 | 179 |
| 2329 | 861 | 355 |
| 2339 | 830 | 0 |
| 2351 | 655 | 0 |
| 2382 | 468 | 0 |
| 2533 | 174 | 0 |
| 2618 | 409 | 0 |
| 2659 | 511 | 0 |
| 2675 | 387 | 0 |
| 2705 | 351 | 0 |
| 2717 | 656 | 0 |
| 2755 | 315 | 0 |
| 2793 | 219 | 0 |
| 2834 | 190 | 0 |
| 2855 | 611 | 0 |
| 2859 | 272 | 0 |
| 2860 | 378 | 0 |
| 2905 | 578 | 0 |
| **total** | **11946** | **3613** |

## Query normalisation (flat cosine, e5-small)

| queries | subclass any@1 | any@3 | any@10 | group any@10 |
|---|---|---|---|---|
| as printed (upper case) | 7.3% | 12.1% | 24.7% | 10.9% |
| lower-cased | 9.3% | 18.0% | 35.3% | 18.2% |
| lower-cased, centred vectors | 2.8% | 8.8% | 23.4% | 9.5% |

Lower-casing is now the default (`normalize_query`). Centring hurts and was dropped.

## Beam and weights (hierarchical search, e5-small, lower-cased)

| config | section @1 / @10 | subclass @1 / @10 | group @1 / @10 | subgroup @1 / @10 |
|---|---|---|---|---|
| beam 3/6/10/25, w .6/.15/.25 | 44.1 / 67.3 | 11.4 / 30.6 | 5.4 / 17.6 | 0.7 / 6.7 |
| beam 5/10/20/40, same w | 43.3 / 71.1 | 11.4 / 32.2 | 5.5 / 17.6 | 0.7 / 7.3 |
| beam 8/20/40/80, same w | 43.4 / 73.4 | 11.4 / 33.6 | 5.5 / 18.7 | 0.7 / 7.6 |
| beam 5/10/20/40, own only | 42.7 / 74.2 | 11.2 / 36.5 | 5.5 / 21.5 | 0.5 / 7.3 |
| beam 5/10/20/40, w .7/.3/0 | **44.6** / 68.9 | **12.3** / 30.3 | **5.9** / 16.6 | **1.0** / 5.5 |
| beam 5/10/20/40, w .4/.3/.3 | 44.6 / 68.7 | 12.3 / 29.4 | 5.9 / 16.1 | 1.0 / 6.1 |

Reading: path support buys a little precision at rank 1, own-similarity alone buys
recall at rank 10; the spread is within the noise of 578 cases. Defaults are now beam
5/10/20/40 and weights own 0.7, path 0.3, subtree 0. The model, not the heuristics, is
the bottleneck for Portuguese titles against an English scheme.

## Baseline with final defaults (`t2ipc eval evals/rpi_2905.jsonl --level subgroup`)

`st:intfloat/multilingual-e5-small`, IPC 20260101 (20270101 is within 1 point):

| level | hit@1 | hit@3 | hit@5 | hit@10 | main@1 | MRR |
|---|---|---|---|---|---|---|
| section | 44.6% | 67.8% | 68.9% | 68.9% | 37.7% | 0.560 |
| class | 22.8% | 42.0% | 43.4% | 43.4% | 18.5% | 0.321 |
| subclass | 12.3% | 27.2% | 30.1% | 30.3% | 8.3% | 0.197 |
| group | 5.9% | 13.8% | 15.6% | 16.6% | 3.8% | 0.099 |
| subgroup | 1.0% | 2.6% | 3.3% | 5.5% | 0.5% | 0.022 |

hit@k plateaus after k=3 at section level because the beam keeps five sections and the
top-10 subgroups come from the first few of them.

## 1,000-case random sample (notebook 01, e5-base, IPC 20260101)

500 cases with abstract and 500 title-only, drawn with seed 0 from all 29 issues.

| scheme | cases | subclass @1 / @10 | group @1 / @10 |
|---|---|---|---|
| PT | abstract | 18.8 / 33.6 | 9.2 / 17.4 |
| EN | abstract | 20.2 / 33.0 | 10.4 / 19.6 |
| PT | title-only | 23.0 / 38.0 | 10.8 / 25.2 |
| EN | title-only | 18.8 / 36.8 | (see notebook) |

Across issues the picture holds: PT scheme ahead on titles, tie on abstracts.

## Model comparison (final defaults, IPC 20260101, `--level subgroup --top-k 10`)

`rpi_2905` = 578 Portuguese titles; `rpi_2100` = 557 Portuguese abstracts.

| model | cases | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 | subgroup @10 |
|---|---|---|---|---|---|---|
| multilingual-e5-small | 2905 titles | 44.6 | 22.8 | 12.3 / 30.3 | 5.9 / 16.6 | 5.5 |
| multilingual-e5-small | 2100 abstracts | 37.0 | 19.6 | 9.9 / 22.4 | 4.1 / 10.1 | 2.7 |
| **multilingual-e5-base** | 2905 titles | 58.7 | 40.1 | 22.8 / 39.3 | 13.0 / 25.1 | 6.9 |
| **multilingual-e5-base** | 2100 abstracts | 53.9 | 34.8 | 19.4 / 31.8 | 7.7 / 16.9 | 3.6 |

e5-base is now the default. Its index takes ~20 minutes to build on an M-series Mac
and 650 MB as CSV, against ~10 minutes and 330 MB for e5-small.

Abstracts score *below* titles with both models. Two likely reasons, not yet
separated: the 2011 abstracts are long narrative prose while titles are dense keyword
lists that resemble IPC titles, and the 2011 gold symbols predate several scheme
revisions. Worth testing: title + abstract concatenated, and gold symbols mapped
through the WIPO concordance list.

## Text style and scheme language (e5-base, IPC 20260101, `--level subgroup --top-k 10`)

EN scheme, default style, is the baseline from the table above.

| index | cases | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 |
|---|---|---|---|---|---|
| EN default | 2905 titles | 58.7 | 40.1 | 22.8 / 39.3 | 13.0 / 25.1 |
| EN `lc` | 2905 titles | 54.8 | 32.0 | 18.3 / 36.3 | 9.7 / 21.6 |
| EN `subclass` | 2905 titles | 50.3 | 31.3 | 16.3 / 33.7 | 6.7 / 17.8 |
| EN `lc-cf-subclass` | 2905 titles | 50.9 | 31.1 | 16.6 / 36.3 | 7.8 / 20.8 |
| **PT (INPI titles)** default | 2905 titles | **62.8** | **41.5** | **27.9 / 40.7** | **14.7 / 25.4** |
| EN default | 2100 abstracts | 53.9 | 34.8 | 19.4 / 31.8 | 7.7 / 16.9 |
| EN `lc` | 2100 abstracts | 47.6 | 27.1 | 15.4 / 28.5 | 6.8 / 15.3 |
| EN `subclass` | 2100 abstracts | 32.5 | 18.0 | 7.9 / 26.6 | 2.3 / 11.7 |
| EN `lc-cf-subclass` | 2100 abstracts | 33.8 | 18.7 | 9.3 / 27.8 | 4.1 / 13.3 |
| **PT (INPI titles)** default | 2100 abstracts | **58.3** | 34.6 | **20.3 / 32.9** | **9.5 / 17.1** |

Reading:

- **Lower-casing the scheme hurts** (4 to 5 points at subclass) even though lower-casing
  the *queries* helps. The scheme's capitalised titles are short and the model copes;
  the query titles were long shouting strings. Keep `lc` off for the scheme.
- **Dropping section and class hurts badly**, worst on abstracts (subclass@1 19.4 to
  7.9). The generic titles are not noise: they anchor the embedding of deep entries.
- **Child-first is neutral** on these models, as expected with no truncation.
- **The Portuguese scheme wins everywhere** for Portuguese queries, by 5 points at
  subclass@1 on titles. This run still had the "[YYYY.MM]" edition tag in 17% of
  titles; the corrected rebuild is reported below.

Decision: the text assembly options were removed from the code; the full top-down
path in its original case is the only assembly (ADR 0006). `--lang PT` is the
recommended index for Portuguese text.

### PT index rebuilt with clean titles (edition tags stripped)

| cases | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 |
|---|---|---|---|---|
| 2905 titles | 63.7 | 42.4 | 27.7 / 40.1 | 14.5 / 26.5 |
| 2100 abstracts | 58.2 | 34.8 | 20.5 / 33.6 | 9.3 / 17.8 |

Within noise of the first PT run: the leftover "[2016.01]" tags were harmless to the
embeddings, but they are gone from the displayed text.

### PT scheme with e5-small (170 MB index, 3 minutes to build)

| cases | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 |
|---|---|---|---|---|
| 2905 titles | 52.2 | 32.4 | 20.6 / 34.6 | 11.2 / 19.9 |
| 2100 abstracts | 49.6 | 23.7 | 14.9 / 24.8 | 6.5 / 11.3 |

The Portuguese scheme lifts e5-small by 8 points at subclass@1 on titles (12.3 to
20.6), more than it lifts e5-base (22.8 to 27.7): the smaller model gains most from
not having to cross languages. e5-small + PT scheme roughly matches e5-base + EN
scheme at a third of the index size.

## Browser demo: int8 vectors (e5-small, IPC 20260101, 1,000-case sample, `--level subgroup --top-k 10`)

The static Space (ADR 0007) ships each vector as int8 with a per-row float32 scale,
a quarter of the size. Same 1,000 random cases as notebook 01 (seed 0, 500 abstracts
and 500 titles), same query vectors, index vectors swapped for their dequantised int8
form. Run on 2026-10-08.

| scheme | vectors | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 | mrr subclass |
|---|---|---|---|---|---|---|
| PT | fp32 | 50.4 | 27.5 | 17.0 / 28.7 | 6.9 / 13.7 | 0.217 |
| PT | int8 | 49.8 | 27.2 | 16.6 / 28.7 | 6.8 / 13.8 | 0.215 |
| EN | fp32 | 38.7 | 21.9 | 12.0 / 25.0 | 5.0 / 12.1 | 0.172 |
| EN | int8 | 38.6 | 22.0 | 12.1 / 24.7 | 5.0 / 12.0 | 0.172 |

Differences of at most 0.4 points in either direction: quantising the vectors is
free. The browser also runs the q8 ONNX embedder instead of the fp32 PyTorch one;
that effect was not measured over the eval set, only checked on a handful of queries
(same top symbols, similarities within 0.01).

### Same measurement for e5-base (the model the Space ships since 2026-10-08)

| scheme | vectors | section @1 | class @1 | subclass @1 / @10 | group @1 / @10 | mrr subclass |
|---|---|---|---|---|---|---|
| PT | fp32 | 56.8 | 34.0 | 20.9 / 35.8 | 10.0 / 21.3 | 0.271 |
| PT | int8 | 56.8 | 34.2 | 21.4 / 36.4 | 10.1 / 21.8 | 0.276 |
| EN | fp32 | 53.5 | 33.9 | 19.5 / 34.9 | 9.7 / 20.3 | 0.257 |
| EN | int8 | 54.0 | 34.0 | 19.3 / 34.3 | 9.5 / 20.3 | 0.254 |

Again within noise (at most 0.9 points, both directions). The q8 ONNX e5-base in the
browser sits further from the PyTorch fp32 vectors than e5-small did (cosine 0.983 to
0.994 against 0.996 to 0.998 on six probe texts), enough to swap near-tied neighbours
such as the 2nd and 3rd result of the fire-hose example; the top result was the same
on every probe.

## Auto level honours `top_k` (e5-base, PT scheme, `--level auto --top-k 10`)

`auto_roots` was fixed at 5, so the auto level never returned more than 5 results
whatever `top_k` said (the browser demo made this visible: Top 5, 8 or 12 gave the same
5 rows). It now defaults to `max(5, top_k)`. hit@1 to hit@5 are unchanged by
construction; hit@10 rises because there are results beyond the fifth.

| cases | section @10 | class @10 | subclass @10 | group @10 | mrr subclass |
|---|---|---|---|---|---|
| 2905 titles, before | 82.4 | 66.1 | 51.6 | 26.8 | 0.363 |
| 2905 titles, after | 87.4 | 76.3 | 62.6 | 31.7 | 0.378 |
| 2100 abstracts, before | 81.7 | 61.4 | 45.8 | 19.2 | 0.297 |
| 2100 abstracts, after | 91.0 | 72.0 | 58.3 | 25.0 | 0.314 |

## Path support and beam, revisited with the PT scheme (1,000-case sample, `--level subgroup --top-k 10`)

The defaults (own 0.7, path 0.3, beam 5/10/20/40) came from sweeps on the English
scheme with e5-small. On the Portuguese scheme, for both models, path support lowers
every number and the beam width changes nothing (default, wide 8/30/60/120 and no
beam agree to 0.3 points). Run on 2026-10-08, not yet adopted as defaults.

| model | weights | subclass @1 / @10 | group @1 / @10 | subgroup @1 / @10 |
|---|---|---|---|---|
| e5-base | own .7 path .3 (current) | 20.9 / 35.8 | 10.0 / 21.3 | 1.8 / 5.4 |
| e5-base | own .85 path .15 | 22.5 / 39.6 | 10.7 / 23.8 | 2.2 / 6.6 |
| e5-base | own 1.0 path 0 | 23.5 / 41.7 | 11.1 / 25.0 | 2.4 / 6.7 |
| e5-small | own .7 path .3 (current) | 17.0 / 28.7 | 6.9 / 13.7 | 1.4 / 3.5 |
| e5-small | own .85 path .15 | 18.3 / 31.6 | 7.4 / 15.8 | 1.6 / 4.2 |
| e5-small | own 1.0 path 0 | 20.1 / 34.1 | 8.4 / 17.0 | 1.7 / 4.6 |

Why it hurts here: the Portuguese section and class titles are generic and close to
every query, so their mean (path support) rewards branches such as A62C (fire
fighting) for texts about antibodies or solar panels; see the hub note in ADR 0007.

### Candidate recall for a reranker (e5-base, PT, flat cosine, `--top-k 100`)

What a second-stage reranker could reach at @1 if it were perfect on the candidates:

| level | hit@1 | hit@10 | hit@25 | hit@50 | hit@100 |
|---|---|---|---|---|---|
| subclass | 23.5 | 59.7 | 64.2 | 64.3 | 64.3 |
| group | 11.1 | 36.2 | 44.7 | 45.7 | 45.7 |
| subgroup | 2.4 | 7.5 | 11.9 | 15.0 | 18.2 |

With the current defaults the same ceilings are 55.6 / 39.7 / 15.3. Recall stops
growing after 25 to 50 candidates at subclass and group level: the candidates for a
reranker are there, the first stage simply orders them badly.

## Path weight set to 0 by default (e5-base, PT scheme, both files, `--top-k 10`)

Adopted on 2026-10-08 after the sweep above: `Weights(own=1.0, path=0.0)` in the
package and in the browser scorer. Before/after on the two standard files, at the
subgroup level and at the auto level.

| cases, level | subclass @1 / @10 | group @1 / @10 | subgroup @1 / @10 | mrr subclass |
|---|---|---|---|---|
| 2905 titles, subgroup, before | 27.7 / 40.1 | 14.5 / 26.5 | 1.6 / 9.3 | 0.330 |
| 2905 titles, subgroup, after | 29.4 / 45.7 | 14.7 / 29.9 | 2.4 / 13.0 | 0.364 |
| 2905 titles, auto, before | 27.7 / 62.6 | 14.7 / 31.7 | 2.4 / 6.9 | 0.378 |
| 2905 titles, auto, after | 28.7 / 65.1 | 14.5 / 32.9 | 2.9 / 7.8 | 0.398 |
| 2100 abstracts, subgroup, before | 20.5 / 33.6 | 9.3 / 17.8 | 0.5 / 1.8 | 0.260 |
| 2100 abstracts, subgroup, after | 23.2 / 38.4 | 11.0 / 20.1 | 0.9 / 2.5 | 0.292 |
| 2100 abstracts, auto, before | 20.5 / 58.3 | 9.5 / 25.0 | 0.9 / 2.9 | 0.314 |
| 2100 abstracts, auto, after | 23.2 / 61.6 | 10.8 / 27.3 | 1.1 / 3.6 | 0.341 |

Every cell but one (group@1 at auto on titles, -0.2) improves; subclass@10 gains 5 to
6 points at the subgroup level. The demo examples still agree with INPI at rank 1.

## Combining the parts of one application (e5-base, PT scheme, 500 cases with title and abstract)

Proxy for the multi-chunk question (the evals hold no full descriptions): the 500
cases of the 1,000-case sample that have both a title and an abstract, `--level
subgroup --top-k 10`, path weight 0. Run on 2026-10-08.

| query | subclass @1 / @10 | group @1 / @10 | subgroup @1 / @10 |
|---|---|---|---|
| abstract only (the eval text) | 22.0 / 38.2 | 10.0 / 20.4 | 1.2 / 4.6 |
| title only | 23.2 / 40.8 | 10.2 / 24.2 | 2.4 / 9.0 |
| title + abstract, one string | 22.0 / 38.6 | 9.8 / 19.4 | 1.2 / 4.0 |
| mean of the title and abstract vectors | 26.6 / 42.6 | 11.8 / 22.6 | 3.0 / 6.8 |
| max per entry over the two vectors | 23.2 / 43.0 | 10.4 / 23.2 | 1.0 / 7.6 |

Concatenating buys nothing; averaging the two vectors adds 4.6 points at subclass@1
over the abstract alone. Adopted: paragraphs are embedded separately and averaged
(`chunking="mean"`), with a paragraph over the token limit cut into sentence chunks.
`chunking="max"` is kept as an option for texts that cover several inventions. The
standard eval files are unaffected (their text is a single paragraph).

## Cross-encoder reranking (bge-reranker-v2-m3 over the top-50 of e5-base, PT scheme, 991 cases)

First stage: default search at the subgroup level with `top_k=50` (path weight 0,
distinct branches). Second stage: `BAAI/bge-reranker-v2-m3` (568M parameters, fp16 on
MPS, 74 pairs per second, `max_length=768`) scores every (query, path text) pair; the
list is reordered and cut to 10. Nine cases of the 1,000-case sample were lost to a
timeout and are excluded from every row. Run on 2026-10-08.

| ordering | subclass @1 / @3 / @10 | group @1 / @3 / @10 | subgroup @1 / @3 / @10 |
|---|---|---|---|
| first stage (cosine) | 23.6 / 38.6 / 43.1 | 11.2 / 21.2 / 26.1 | 2.4 / 4.2 / 7.6 |
| cross-encoder only | 32.3 / 45.3 / 47.9 | 16.0 / 27.2 / 30.6 | 3.0 / 6.1 / 9.7 |
| score × (0.5 + 0.5 · sigmoid(ce)) | 31.6 / 45.6 / 48.2 | 16.6 / 27.2 / 30.5 | 3.3 / 6.2 / 9.7 |
| score × sigmoid(ce) | 32.2 / 45.3 / 47.9 | 16.2 / 27.3 / 30.5 | 3.0 / 6.1 / 9.7 |
| cross-encoder as tie-break only | 28.8 / 43.5 / 47.3 | 14.6 / 24.9 / 29.4 | 3.0 / 5.2 / 8.8 |
| reciprocal rank fusion | 29.2 / 44.9 / 48.4 | 14.9 / 26.1 / 30.7 | 3.3 / 5.5 / 9.8 |

Titles gain more than abstracts (subclass@1 25.0 to 35.3 on titles, 22.2 to 29.3 on
abstracts). Multiplying the first-stage score by the judge works, as proposed; the
`0.5 + 0.5 · sigmoid` form keeps the cosine score meaningful and is the most even
across levels, so it is the default fusion. The candidate ceiling (hit@50 of 64% at
subclass) is far from reached: the reranker is good, not perfect.

### The browser candidate: jina-reranker-v2-base-multilingual in ONNX, 8-bit (same 300 cases)

`jinaai/jina-reranker-v2-base-multilingual` (278M parameters) scored the same
candidate lists through its 8-bit ONNX export on CPU (24 pairs per second with
onnxruntime, `max_length=768`); the bge rows are the first 300 cases of the run above.

| reranker, ordering | subclass @1 / @3 / @10 | group @1 / @3 / @10 | subgroup @1 / @3 / @10 |
|---|---|---|---|
| none (first stage) | 27.0 / 39.3 / 43.0 | 14.7 / 23.3 / 27.0 | 4.0 / 6.0 / 7.3 |
| bge-reranker-v2-m3, blend | 31.7 / 46.3 / 48.7 | 17.7 / 27.3 / 30.7 | 4.3 / 6.3 / 9.0 |
| jina-reranker-v2 q8, blend | 32.3 / 43.7 / 47.0 | 18.0 / 26.0 / 29.7 | 3.0 / 6.3 / 10.7 |
| jina-reranker-v2 q8, judge only | 31.0 / 44.3 / 46.3 | 16.3 / 26.0 / 29.0 | 2.0 / 5.7 / 10.7 |

Equal within noise at @1, a point or two behind at @3 and @10, at half the size and
in a format a browser runs: the Space ships jina in 8 bits as an opt-in second stage.
The package keeps bge-reranker-v2-m3 as its default (its PyTorch code loads with the
current transformers; jina's remote code does not).
