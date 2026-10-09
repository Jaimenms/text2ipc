# text2ipc

Map free text, typically a patent abstract, to a ranked list of International Patent
Classification (IPC) symbols. Every IPC entry is embedded once from its full path
(section > class > subclass > group > subgroups), a query of any length is embedded by
paragraph and sentence chunks, and an optional cross-encoder re-judges the best
candidates. Portuguese first; English and French schemes built in.

Study repository: the package is meant for PyPI, the docs record every decision and
every measurement along the way.

## Quick start

Try it without installing anything:
[huggingface.co/spaces/jaimenms/text2ipc](https://huggingface.co/spaces/jaimenms/text2ipc)
runs the embedder and the scoring in your browser (ADR 0007).

Use the published Portuguese index (IPC 2026, `intfloat/multilingual-e5-base`):

```bash
pip install "text2ipc[st] @ git+https://github.com/Jaimenms/text2ipc"
t2ipc download                     # index + scheme from huggingface.co/jaimenms/text2ipc-pt
t2ipc classify "Aparelho para combate a incêndios com mangueira flexível" --lang PT --level group
t2ipc classify "..." --lang PT --level group --rerank    # + cross-encoder second stage
```

```python
from text2ipc import IpcClassifier

clf = IpcClassifier("20260101", lang="PT")
for m in clf.classify("Aparelho para combate a incêndios com mangueira flexível", level="group"):
    print(m.pretty, round(m.score, 3), m.text)
```

```
A62C 33/00  0.861  NECESSIDADES HUMANAS > SALVAMENTO; COMBATE AO FOGO > COMBATE AO FOGO > Acessórios para mangueiras
A62C 15/00  0.853  NECESSIDADES HUMANAS > SALVAMENTO; COMBATE AO FOGO > COMBATE AO FOGO > ... > Extintores essencialmente do tipo mochila
```

The same index runs as a Hugging Face Inference Endpoint (JSON in, JSON out) from the
repository [jaimenms/text2ipc-pt](https://huggingface.co/jaimenms/text2ipc-pt); the
French scheme is published the same way as
[jaimenms/text2ipc-fr](https://huggingface.co/jaimenms/text2ipc-fr) (`t2ipc download
jaimenms/text2ipc-fr`, then `--lang FR`). Scheme titles in the language of the text
score best, so pick the scheme that matches the query.

## Documentation

| Document | What it answers |
|---|---|
| [docs/usage.md](docs/usage.md) | How to use it: hosted endpoint, local install with a prebuilt index, building your own index, evaluating, publishing |
| [docs/methodology.md](docs/methodology.md) | How it works: scheme parsing, path texts, the hierarchical scoring heuristics, languages, evaluation protocol |
| [docs/evals.md](docs/evals.md) | What it scores: every eval run with its numbers, including the experiments that were rejected |
| [docs/adr/](docs/adr/) | Why it is built this way: one record per decision that is expensive to reverse (path text, Parquet tables, languages, eval source, distribution) |
| [notebooks/01_text2ipc.ipynb](notebooks/01_text2ipc.ipynb) | Walk-through of the interface and an evaluation on 1,000 INPI applications, with outputs |
| [notebooks/02_rerank.ipynb](notebooks/02_rerank.ipynb) | The second stage: cross-encoder verdicts, fusion rules, one judged run evaluated several ways |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each version, with the measurements behind it |
| [CLAUDE.md](CLAUDE.md) | Conventions for contributors and coding agents |

## Repository layout

```
.
├── README.md                     this file
├── CHANGELOG.md                  per-version changes; releases are tags here and on the Hub
├── CLAUDE.md                     conventions, layout, commands
├── pyproject.toml                package metadata; `uv sync --all-extras` installs everything
├── src/text2ipc/
│   ├── __init__.py               public API: IpcClassifier, classify, Match
│   ├── classifier.py             loads one index + one embedder, answers queries
│   ├── cli.py                    the `t2ipc` command line
│   ├── config.py                 levels, default model, home directory resolution
│   ├── versions.py               IPC versions published by WIPO (latest / current)
│   ├── textnorm.py               query normalisation (lower-case shouting titles)
│   ├── scheme/                   the IPC scheme as data
│   │   ├── download.py           WIPO master files (EN, FR) -> data/wipo/
│   │   ├── parse.py              XML -> entries, depth-first, headings attached to groups
│   │   ├── inpi.py               Portuguese titles from INPI's IPC Publication JSON
│   │   ├── overlay.py            apply a symbol,title translation table
│   │   ├── table.py              SchemeTable: symbol paths, rendered texts, Parquet I/O
│   │   ├── model.py              IpcNode
│   │   └── symbols.py            A01B0001020000 <-> "A01B 1/02", levels
│   ├── embeddings/               Embedder protocol + backends
│   │   ├── st.py                 sentence-transformers (default)
│   │   ├── ollama.py             any Ollama embedding model
│   │   └── hashing.py            deterministic bag-of-words, for tests
│   ├── index/                    vectors per (version, language, model)
│   │   ├── store.py              IpcIndex: Parquet index table + hierarchy arrays
│   │   ├── build.py              embed, reusing unchanged vectors from the previous version
│   │   ├── paths.py              file naming and discovery
│   │   ├── publish.py            `t2ipc download` from the Hugging Face Hub
│   │   └── migrate.py            legacy CSV -> Parquet, hash-verified
│   ├── search/scorer.py          flat cosine + path/subtree support, beam descent,
│   │                             auto level, distinct-branch post-processing
│   ├── chunking.py               paragraphs and sentence chunks for texts over the token limit
│   ├── rerank/                   Reranker protocol; cross-encoder (bge-reranker-v2-m3) and hash backends
│   ├── eval/                     cases, RPI parsers (XML and legacy text), hit@k harness
│   ├── hf/                       Inference Endpoints handler and repository export
│   └── web/                      static Space export; static/scorer.js is the port of scorer.py
├── scripts/
│   ├── publish_hf.sh             export + upload an index to the Hub (endpoint repo)
│   ├── publish_space.sh          export + upload the browser demo (static Space)
│   ├── make_notebooks.py         generates notebooks/ (edit this, not the .ipynb)
│   └── build_index.py            thin wrapper over `t2ipc build`
├── notebooks/                    executed study notebooks: 01 interface + eval, 02 reranker
├── evals/rpi_<issue>.jsonl       29 INPI issues, ~12k applications with their IPC symbols
├── tests/                        46 tests; no network, no real model (hash embedder + fixtures);
│                                 js_parity.mjs runs scorer.js under Node against the Python scorer
├── docs/                         usage, methodology, evals, adr/
└── data/                         (gitignored) wipo/ inpi/ rpi/ raw sources, scheme/ index/ Parquet tables
```

## Development

```bash
uv sync --all-extras
uv run t2ipc build --version 20260101 --lang PT      # ~3 min with e5-small, ~10 with e5-base
uv run t2ipc eval evals/rpi_2905.jsonl --lang PT --level subgroup
uv run pytest && uv run ruff check . && uv run ruff format .
uv run python scripts/make_notebooks.py && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
scripts/publish_hf.sh --lang PT                      # endpoint repo on the Hub
scripts/publish_space.sh                             # browser demo (static Space)
```

License: [MIT](LICENSE). Use, copy and modify freely, keeping the copyright notice
that names the author.

Data sources: IPC master files by [WIPO](https://www.wipo.int/classifications/ipc/en/),
Portuguese titles by [INPI Brazil](https://ipc.inpi.gov.br/classifications/ipc/ipcpub/),
evaluation cases from INPI's weekly *Revista da Propriedade Industrial*.
