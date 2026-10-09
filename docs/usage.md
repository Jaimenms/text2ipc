# Using text2ipc

Four ways to get an IPC suggestion for a text, from lightest to heaviest: open the
browser demo, call the hosted endpoint, run the package locally with a prebuilt index,
or build your own index.

## 1. In the browser: Hugging Face Space

[huggingface.co/spaces/jaimenms/text2ipc](https://huggingface.co/spaces/jaimenms/text2ipc)
is a static page: it downloads the quantised embedder (`Xenova/multilingual-e5-base`,
279 MB) and the IPC index (about 70 MB per language) once, then embeds and scores in
the browser. Nothing is sent to a server. Pick the scheme language (PT or EN), the
level (`auto`, section ... subgroup) and how many results; `?q=...&lang=PT&level=group`
in the URL pre-fills and runs a query. The results are drawn as one tree, their paths
merged under a root node: nodes are the symbol parts (IPC › A › 62 › C › 25/00 › 25/01), each edge carries the
similarity of that entry to the text, and the edge into a result carries its score.
Hovering or focusing an edge or node shows the entry's title. The table below the
tree lists the same results with their full path text. "Rerank" loads a second
model once (`jinaai/jina-reranker-v2-base-multilingual`, 280 MB in 8 bits) and
re-judges the top 25 candidates against the text; it takes some seconds per query
in the browser and adds a Judge column. That model is licensed CC BY-NC 4.0
(non-commercial), which suits a demo; the package's default reranker
(`BAAI/bge-reranker-v2-m3`) is Apache-2.0. The example buttons are real
applications from `evals/demo_examples.jsonl` (title plus abstract, with the symbols
INPI assigned); they were chosen so that the top group on the PT scheme agrees with
INPI, and results that agree at their level are marked ✓. Results differ slightly from the package: the
vectors are int8 and the model is 8-bit (ADR 0007, numbers in `docs/evals.md`).

## 2. Hosted: Hugging Face Inference Endpoint

The repository `jaimenms/text2ipc-pt` holds the Portuguese IPC 2026 index and a
`handler.py`. Deployed as an Inference Endpoint it answers JSON:

```bash
curl -X POST "$ENDPOINT_URL" \
  -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" \
  -d '{"inputs": "Aparelho para combate a incêndios com mangueira flexível reforçada",
       "parameters": {"level": "group", "top_k": 5}}'
```

```python
import requests

r = requests.post(
    ENDPOINT_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={"inputs": "...", "parameters": {"level": "subclass", "top_k": 5}},
)
for m in r.json():
    print(m["symbol"], m["score"], m["path"])
```

`inputs` may be a string or a list of strings (one result list per input). Parameters:
`level` (section, class, subclass, group, subgroup, auto), `top_k`, `gap`,
`auto_margin`, `normalize`, `chunking` (`mean`, `max`, `truncate`), `rerank` (true, or
a reranker spec), `candidates`, `fusion`. Each result has `symbol`, `canonical`,
`level`, `depth`, `score`, `similarity`, `judge` (0..1 when reranked, else null),
`title` and `path` (the full section-to-entry text).

## 3. Local, with a prebuilt index

Python 3.11 to 3.13. Install the package with the sentence-transformers extra:

```bash
pip install "text2ipc[st]"            # once on PyPI
# until then, from the repository:
pip install "text2ipc[st] @ git+https://github.com/Jaimenms/text2ipc"
```

Fetch the index and scheme table (about 230 MB) from the Hub, then classify:

```bash
t2ipc download                          # jaimenms/text2ipc-pt by default
t2ipc classify "Aparelho para combate a incêndios com mangueira flexível" --lang PT --level group
t2ipc classify "..." --lang PT --level auto --json     # machine-readable
t2ipc show "A62C 15/00" --lang PT                       # the path an entry was embedded with
```

The first call downloads the embedding model (`intfloat/multilingual-e5-base`, about
1.1 GB) from the Hub and caches it. Files land in `~/.cache/text2ipc` (or
`$TEXT2IPC_HOME`).

Python:

```python
from text2ipc import IpcClassifier

clf = IpcClassifier("20260101", lang="PT")
for m in clf.classify(
    "Aparelho para combate a incêndios com mangueira flexível", level="group", top_k=5
):
    print(m.pretty, round(m.score, 3), m.text)
```

`level="auto"` descends as far as the evidence supports. Results are distinct
branches of the IPC tree: an ancestor or descendant of a better-ranked result is
dropped (`dedupe_branches=False` on `SearchParams` keeps them).

`rerank=True` adds a second stage: a multilingual cross-encoder
(`BAAI/bge-reranker-v2-m3`, 2.2 GB, downloaded on first use) judges the top
candidates against the text and the scores are fused; each match then has a
`judge` value in 0..1. It costs under a second per query on an Apple GPU and raises
subclass@1 by 8 points on the evals (`docs/evals.md`):

```bash
t2ipc classify "..." --lang PT --level group --rerank
t2ipc classify "..." --lang PT --rerank --reranker ce:BAAI/bge-reranker-v2-m3 --candidates 50
t2ipc eval evals/rpi_2905.jsonl --lang PT --rerank
```

Any length of text works: a whole description is cut into sentence chunks that fit
the embedder and the chunk vectors are averaged (`chunking="mean"`; `"max"` scores
an entry by its best chunk, `"truncate"` keeps only what the embedder can take).
`clf.last_chunks` says how many chunks the last call used; the CLI prints it in the
table title. The browser demo does the same.

Run the exact handler the endpoint runs, locally:

```python
import sys

sys.path.insert(0, "hf/text2ipc-pt")  # a folder made by `t2ipc hf-export`
from handler import EndpointHandler

h = EndpointHandler("hf/text2ipc-pt")
h({"inputs": "...", "parameters": {"level": "group", "top_k": 5}})
```

## 4. Local, building your own index

```bash
uv sync --all-extras
uv run t2ipc versions                       # IPC versions published by WIPO
uv run t2ipc build --version 20260101 --lang PT     # Portuguese titles from INPI (~3 min with e5-small)
uv run t2ipc build --version 20260101 --lang EN --model st:intfloat/multilingual-e5-small
uv run t2ipc build --version 20270101 --lang PT     # reuses unchanged vectors from 20260101
```

Languages: `EN` and `FR` come from WIPO master files, `PT` from INPI Brazil's
translation; any other language can be supplied as a `--titles-csv symbol,title`.
Models: any sentence-transformers model (`st:<name>`), any Ollama embedding model
(`ollama:<name>`). Building takes 3 to 10 minutes on an Apple M-series depending on
the model; the result is two Parquet tables under the home directory.

## 5. Evaluate

```bash
uv run t2ipc rpi 2905                        # cases from an INPI RPI issue
uv run t2ipc eval evals/rpi_2905.jsonl --lang PT --level subgroup --show-misses 5
```

`notebooks/01_text2ipc.ipynb` walks through all of the above and reports accuracy on
1,000 INPI applications. Current numbers and their history are in `docs/evals.md`.

## 6. Publish

```bash
scripts/publish_hf.sh --lang PT                # endpoint repo: export + upload to <you>/text2ipc-pt
scripts/publish_hf.sh --lang EN --public       # another language, public repo
scripts/publish_space.sh                       # browser demo: PT + EN, e5-small, <you>/text2ipc
scripts/publish_space.sh --lang PT --private   # one language, private Space
```

Needs `hf auth login` once. Both scripts are idempotent: unchanged files are skipped.
The Space script runs `t2ipc web-export`, which needs an index built with the same
model for every language it ships (e5-base by default: `t2ipc build --lang PT`);
pass `--model` to ship another embedder that has an ONNX twin on the Hub, for example
e5-small at 118 MB in the browser.
