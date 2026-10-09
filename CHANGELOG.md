# Changelog

All notable changes to `text2ipc`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Each release is also a tag on the Hub
repositories (`jaimenms/text2ipc-pt`, `spaces/jaimenms/text2ipc`), so
`t2ipc download --revision v0.2.0` pins an edition. Decisions that are expensive to
reverse have their own record under `docs/adr/`; every measurement is in
`docs/evals.md`.

## [Unreleased]

## [0.2.2] - 2026-10-09

### Fixed
- `rerank=True` failed with `TypeError: unexpected keyword argument 'model_kwargs'`
  on sentence-transformers 3.x, whose CrossEncoder still used `automodel_args` and
  `activation_fct`. The backend now reads the installed signatures and works with
  3.x and with 4.0 or later (tested with 3.4.1 and 6.0.1).

### Added
- `notebooks/02_rerank.ipynb`: the second stage step by step, with one application
  before and after, the judge's verdicts, and one judged run evaluated under every
  fusion rule (`text2ipc.eval.judge_candidates` and `fusion_results`).

### Removed
- Browser demo: no comparison with the office's symbols. The "INPI assigned" line,
  the ✓ marks and the symbols shipped with the examples are gone; the example
  buttons only fill the box with a title and an abstract. Evals stay in validation.

### Changed
- Notebook 01 describes the current scoring (cosine by default, paragraphs averaged)
  and points at notebook 02 instead of listing the reranker as future work.
- Docs: the methodology's evaluation section states the current numbers and the
  second-stage eval helpers; the usage guide covers both notebooks and the version
  tags; the eval log opens with the state on 2026-10-09 and a map of its sections;
  ADR 0005 is amended with the fixed repository names and the tagging rule.

## [0.2.1] - 2026-10-09

### Changed
- Browser demo: the reranker runs in a Web Worker with a progress bar for the model
  download and for the judging, pair by pair; the page stays responsive.
- Browser demo: nothing runs by itself. Example buttons and `?q=` links fill the
  text box; only Classify starts a classification (the Cmd/Ctrl+Enter shortcut is
  gone too).

### Fixed
- The model card generated for the Hub repository (`t2ipc hf-export`) documents the
  endpoint parameters added in 0.2.0 (`rerank`, `candidates`, `fusion`, `chunking`),
  the `judge` field of the response, the reranker model and the version tags; both
  the embedder and the reranker are listed as `base_model` so they show in the
  Hub's model tree.

## [0.2.0] - 2026-10-09

### Added
- Browser demo as a free static Hugging Face Space (ADR 0007):
  `t2ipc web-export` writes a manifest, the scheme as columnar JSON, int8 vectors
  and the page; `scorer.js` is a port of `search/scorer.py` kept in step by a Node
  parity test; `scripts/publish_space.sh` publishes it. Live at
  https://huggingface.co/spaces/jaimenms/text2ipc.
- Results drawn as one tree under a root node, with each ancestor's similarity on
  its edge, the score on the edge into a result, and the entry title on hover or
  keyboard focus.
- Second stage: `rerank=True` (CLI `--rerank`, endpoint `rerank`) keeps
  `max(5 * top_k, 25)` candidates and judges each (text, path text) pair with
  `BAAI/bge-reranker-v2-m3`, fused as `cosine * (0.5 + 0.5 * sigmoid(logit))`;
  `Match.judge` carries the verdict. The Space offers the same with
  `jina-reranker-v2-base-multilingual` in 8 bits (CC BY-NC 4.0, demo only).
- Texts of any length: paragraphs are embedded separately and averaged, a
  paragraph over the embedder's token limit is cut into sentence chunks
  (`chunking.py`); `chunking="max"` scores an entry by its best chunk and
  `search` accepts a stack of query vectors. Embedders expose `max_tokens`,
  `count_tokens` and `embed_queries`.
- Demo examples are real INPI applications (`evals/demo_examples.jsonl`, title
  plus abstract plus the office's symbols) chosen so the top group agrees with
  the office; agreeing results are marked.
- Every Hub publish is tagged `v<version>` (`scripts/hf_tag.sh`).
- `LICENSE` (MIT) and this changelog.

### Changed
- `Weights` default to `own=1.0, path=0.0`: on the Portuguese scheme path support
  lowered every hit rate for both e5 models (ADR 0008). Subclass@1 on the 2100
  abstracts goes from 20.5% to 23.2%, @10 from 33.6% to 38.4%.
- Query normalisation runs per paragraph, so a shouting title above a mixed-case
  abstract is still lower-cased.
- The auto level honours `top_k`: `auto_roots` defaults to `max(5, top_k)`
  instead of a fixed 5.
- `publish_hf.sh` keeps the repository name `text2ipc-pt` and tags each upload.

### Fixed
- Browser demo: a click during the model download no longer starts a second
  download; classifications run one at a time; "Ready" is shown only when every
  load has finished; hovering a score label reaches the edge beneath it.

### Measured
- Cross-encoder over the top 50: subclass@1 23.6% to 31.6%, group@1 11.2% to
  16.6% on 991 cases; jina in 8 bits equals bge at @1 on 300 cases.
- Mean of the title and abstract vectors: subclass@1 26.6% against 22.0% for the
  abstract alone and 22.0% for the two concatenated (500 cases).
- int8 vectors cost at most 0.9 points at any level; the 8-bit ONNX e5-base in
  the browser keeps the same top result on every probe text.

## [0.1.0] - 2026-09-09

### Added
- The package: `classify(text, version, level)`, `IpcClassifier` and the `t2ipc`
  command line (`versions`, `build`, `scheme`, `classify`, `show`, `rpi`, `eval`,
  `download`, `hf-export`, `migrate-csv`).
- IPC scheme from WIPO master files (EN, FR) with Portuguese titles overlaid from
  INPI's IPC Publication; every entry embedded once from its full ancestor path
  (ADR 0001, 0003).
- Hierarchical scoring: flat cosine plus path support, subtree support, beam
  descent, auto level, gap and distinct-branch post-processing.
- Indexes per IPC version that reuse unchanged vectors from the previous
  version; Parquet scheme and index tables (ADR 0006, superseding the CSV layout
  of ADR 0002).
- Evaluation from INPI's Revista da Propriedade Industrial: 29 issues, about
  12,000 applications with their office-assigned symbols (ADR 0004); hit@k harness.
- Distribution through the Hugging Face Hub: `t2ipc download` and an Inference
  Endpoints handler (ADR 0005), published as `jaimenms/text2ipc-pt`.
- Embedding backends: sentence-transformers (default `intfloat/multilingual-e5-base`),
  Ollama, and a hash embedder for tests.
- Study notebook and documentation: usage, methodology, eval log, ADRs.

[Unreleased]: https://github.com/Jaimenms/text2ipc/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/Jaimenms/text2ipc/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Jaimenms/text2ipc/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Jaimenms/text2ipc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jaimenms/text2ipc/releases/tag/v0.1.0
