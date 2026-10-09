# 0007 Browser demo as a static Hugging Face Space

Status: accepted, 2026-10-08

## Context

A live demonstration should cost nothing to keep up. On 2026-10-08 Hugging Face Spaces
that run on compute (Gradio or Docker SDK) require a paid plan to create; static
Spaces stay free for everyone. ZeroGPU Gradio Spaces are free for personal accounts in
good standing, but tie the demo to GPU quotas and account conditions, and the
Inference Endpoint of ADR 0005 bills per hour. The classifier itself is small: a
query embedding plus a dot product against 80k vectors and a few passes over the
tree. Everything it needs can run in a browser.

## Decision

`t2ipc web-export <dir>` assembles a static Space that classifies client side:

- **Vectors** go out as int8 with one float32 scale per row (`data/<lang>/vectors.bin`),
  62 MB per language for e5-base instead of 246 MB (31 instead of 123 for e5-small).
  Cosine between a dequantised vector and the original stays above 0.999.
- **Scheme** goes out as columns `symbol, level, depth, parent, title, heading`
  (`data/<lang>/scheme.json`, 7 to 8 MB); the page renders the path text from the
  titles, exactly as `SchemeTable.text_at` does.
- **Embedder**: `transformers.js` loads the ONNX twin of the sentence-transformers
  model from the Hub (`Xenova/multilingual-e5-small`, `q8`, 118 MB), mean pooling and
  L2 normalisation as in the Python backend, with the same `query: ` prefix.
  The Space first shipped e5-small (118 MB) for its size; it was switched to e5-base
  (279 MB) the same day, when the small model put the fire-hose entries of A62C at
  the top of unrelated texts: its similarities are compressed (top 0.880, 100th 0.873)
  and the generic ancestors of A62C collect path support. e5-base is also 7 points
  better at subclass@1 on the evals. `--model` still selects either.
- **Scoring**: `web/static/scorer.js` is a line-by-line port of `search/scorer.py`
  (path support, subtree support, beam descent, auto level, branch de-duplication).
  `tests/test_web.py` exports the mini scheme with float32 vectors, runs the port
  under Node and requires the same symbols in the same order, scores within 1e-5.
- One Space can carry several languages on one embedder (`--lang PT --lang EN`);
  the page loads an index when it is selected. `scripts/publish_space.sh` creates
  the Space and uploads with `hf upload --repo-type space`.

## Consequences

- The scorer exists twice. A change to `scorer.py` is not done until `scorer.js`
  matches and the parity test covers the new behaviour.
- Results in the browser differ slightly from the Python package: int8 vectors and
  the q8 model shift similarities in the third decimal, which can swap neighbours
  with near-equal scores. The measured effect is recorded in `docs/evals.md`.
- A first visit downloads about 350 MB with e5-base (model, vectors, scheme), about
  160 MB with e5-small; browsers cache the model and the data files, so later visits
  are quick. No server, no sleep, no cost,
  and the text never leaves the visitor's machine.
- What the Space publishes is derived data only: scheme titles (WIPO, INPI) and
  quantised vectors. Raw sources and eval issues stay out, as in ADR 0005.
