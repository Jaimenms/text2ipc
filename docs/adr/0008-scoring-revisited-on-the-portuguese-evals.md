# 0008 Scoring revisited on the Portuguese evals: cosine, paragraph mean, cross-encoder

Status: accepted, 2026-10-08

## Context

The hierarchy heuristics of ADR 0001 and `docs/methodology.md` were tuned on the
English scheme with e5-small. With the Portuguese scheme and e5-base in place, and
with real applications as examples in the browser demo, three things showed up on
the 1,000-case eval sample (`docs/evals.md`, 2026-10-08):

1. Path support lowers every hit rate for both e5 models: Portuguese section and
   class titles are generic and close to every query, so bland ancestors (A62C,
   fire fighting) collect support for texts about anything.
2. The mean of the title vector and the abstract vector beats the abstract alone by
   4.6 points at subclass@1, while concatenating the two strings gains nothing.
3. The first stage orders candidates far worse than it misses them: the office's
   subclass is among the top 50 in 64% of cases but first in 24%.

## Decision

- `Weights` default to `own=1.0, path=0.0`; path and subtree support stay as
  parameters. The beam stays (it changes nothing measurable and costs nothing).
- A query is embedded by paragraph: blank-line separated parts are embedded
  separately and averaged (`chunking="mean"`), and a paragraph over the embedder's
  token limit is cut into sentence chunks (`chunking.split_text`). Normalisation
  runs per paragraph. `chunking="max"` (best chunk per entry) and `"truncate"` are
  options; `search` accepts a stack of query vectors.
- A second stage is available: `rerank=True` keeps `max(5 * top_k, 25)` candidates,
  judges each (text, path text) pair with `BAAI/bge-reranker-v2-m3` and fuses with
  `cosine * (0.5 + 0.5 * sigmoid(logit))`. Measured: subclass@1 23.6% to 31.6%,
  group@1 11.2% to 16.6% on 991 cases. It is opt-in in the package (2.2 GB model,
  about 0.7 s per query on an Apple GPU) and exposed on the CLI, the endpoint
  handler and the eval command.
- The browser scorer (`web/static/scorer.js`) carries the same defaults, the same
  chunking and the same multi-vector search; the parity test covers both.

## Consequences

- Hit rates on the standard files move up at every level (tables in
  `docs/evals.md`); the notebook was re-executed with the new defaults.
- The eval protocol is unchanged: the eval text stays the abstract alone, so the
  paragraph policy is measured on the 500 cases that have both parts, not on the
  standard files.
- The reranker is the first component that reads the query and an entry together;
  it does not change the index, so published indexes stay valid. Any reranker with
  a `score(query, texts)` method can be plugged in (`rerank.Reranker`).
- The browser demo ships `jina-reranker-v2-base-multilingual` in 8 bits (280 MB)
  as an opt-in: it equals bge-reranker-v2-m3 at @1 on 300 cases at half the size,
  and its ONNX export runs in transformers.js. It is licensed CC BY-NC 4.0, so it
  stays confined to the non-commercial demo; the package default is Apache-2.0.
