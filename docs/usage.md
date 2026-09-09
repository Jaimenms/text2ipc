# Using text2ipc

Three ways to get an IPC suggestion for a text, from lightest to heaviest: call the
hosted endpoint, run the package locally with a prebuilt index, or build your own index.

## 1. Hosted: Hugging Face Inference Endpoint

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
`auto_margin`, `normalize`. Each result has `symbol`, `canonical`, `level`, `depth`,
`score`, `similarity`, `title` and `path` (the full section-to-entry text).

## 2. Local, with a prebuilt index

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

Run the exact handler the endpoint runs, locally:

```python
import sys

sys.path.insert(0, "hf/text2ipc-pt")  # a folder made by `t2ipc hf-export`
from handler import EndpointHandler

h = EndpointHandler("hf/text2ipc-pt")
h({"inputs": "...", "parameters": {"level": "group", "top_k": 5}})
```

## 3. Local, building your own index

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

## 4. Evaluate

```bash
uv run t2ipc rpi 2905                        # cases from an INPI RPI issue
uv run t2ipc eval evals/rpi_2905.jsonl --lang PT --level subgroup --show-misses 5
```

`notebooks/01_text2ipc.ipynb` walks through all of the above and reports accuracy on
1,000 INPI applications. Current numbers and their history are in `docs/evals.md`.

## 5. Publish

```bash
scripts/publish_hf.sh --lang PT                # export + upload to <you>/text2ipc-pt
scripts/publish_hf.sh --lang EN --public       # another language, public repo
```

Needs `hf auth login` once. The script is idempotent: unchanged files are skipped.
