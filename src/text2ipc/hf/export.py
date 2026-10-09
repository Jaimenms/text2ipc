"""Assemble a Hugging Face model repository directory for Inference Endpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import __version__ as package_version
from ..config import DEFAULT_RERANKER, home
from ..index import IpcIndex
from ..index.paths import available_indexes, scheme_table_path

REQUIREMENTS = """numpy>=1.26
httpx>=0.27
typer>=0.12
rich>=13.7
sentence-transformers>=3.0
"""

#: Example texts per scheme language, so each repository's card speaks its language.
CARD_EXAMPLES: dict[str, tuple[str, str]] = {
    "PT": (
        "enxada manual com duas lâminas para capina",
        "Aparelho para combate a incêndios com mangueira flexível",
    ),
    "EN": (
        "hand hoe with two blades for weeding",
        "Fire-fighting apparatus with a flexible hose",
    ),
    "FR": (
        "houe à main à deux lames pour le désherbage",
        "Appareil de lutte contre l'incendie avec tuyau flexible",
    ),
}

MODEL_CARD = """---
license: mit
language:
  - {lang_tag}
pipeline_tag: text-classification
tags:
  - patents
  - ipc
  - classification
  - embeddings
  - reranking
  - text2ipc
base_model:
  - {embedder_model}
  - {reranker_model}
base_model_relation: merge
---

# text2ipc {package_version}: IPC {version} ({lang}) via {embedder_model}

Both models above are listed as base models so that they show in the model tree
(the Hub only has "merge" as a relation for two models): the index in this
repository derives from the first, and the second is downloaded and called at run
time when reranking is on. No weights are merged.

Maps free text (a patent abstract, a title, or a whole description) to ranked
International Patent Classification symbols. Every IPC entry is embedded once from its
full ancestor path (section > class > subclass > group > subgroups); a query is
embedded by paragraph (title and abstract separately, averaged), a paragraph over the
embedder's limit is cut into sentence chunks, and the entries are ranked by cosine
similarity, walking the hierarchy to answer at the level asked. An optional second
stage, the **reranker**, re-judges the best candidates with a cross-encoder.

This repository is a **custom Inference Endpoints handler** (`handler.py`). Deploy it as
an Inference Endpoint and call:

```json
{{"inputs": "{example_input}",
 "parameters": {{"level": "group", "top_k": 5, "rerank": true}}}}
```

`inputs` is a string or a list of strings (one result list each). Parameters:

| parameter | values | default | meaning |
|---|---|---|---|
| `level` | section, class, subclass, group, subgroup, auto | subgroup | where in the hierarchy to answer; `auto` descends while the evidence supports it |
| `top_k` | integer | 10 | results per input (distinct branches, so possibly fewer) |
| `rerank` | true, false, or a spec such as `ce:BAAI/bge-reranker-v2-m3` | false | second stage: judge the candidates with a cross-encoder |
| `candidates` | integer | max(5 × top_k, 25) | how many first-stage entries the reranker judges |
| `fusion` | blend, judge, product | blend | `blend` is `cosine × (0.5 + 0.5 × sigmoid(logit))` |
| `chunking` | mean, max, truncate | mean | how the paragraphs and chunks of a long text combine |
| `gap` | float | none | drop results more than this below the best first-stage score |
| `auto_margin` | float | 0.02 | `auto` level: keep descending while a child is within this of its parent |
| `normalize` | true, false | true | collapse whitespace and lower-case shouting paragraphs |

The response is a list of `{{symbol, canonical, level, depth, score, similarity, judge,
title, path}}`: `similarity` is the cosine, `score` the value the list is ordered by,
`judge` the reranker's verdict in 0..1 (null without `rerank`), `path` the full
section-to-entry text.

**Reranker.** `rerank: true` loads `{reranker_model}` (568M parameters, Apache-2.0,
about 2.2 GB) on first use and judges each (text, path text) pair; on 991 INPI
applications it lifts subclass@1 from 23.6% to 31.6% and group@1 from 11.2% to
16.6%, for well under a second per input on a GPU. Size the endpoint for the extra
model when you enable it.

## Use it locally

```bash
pip install "text2ipc[st] @ git+https://github.com/Jaimenms/text2ipc"
t2ipc download {repo_id}                       # or --revision v{package_version} to pin this edition
t2ipc classify "{example_text}" --lang {lang} --level group
t2ipc classify "..." --lang {lang} --level group --rerank
```

```python
from text2ipc import IpcClassifier
clf = IpcClassifier("{version}", lang="{lang}")
for m in clf.classify("...", level="group", top_k=5, rerank=True):
    print(m.pretty, round(m.score, 3), m.judge, m.text)
```

## Versions

Every publication of this repository is tagged `v<package version>` (`v0.2.0`,
`v0.2.0-2`, ...), so `t2ipc download {repo_id} --revision <tag>` and the Hub's
`revision` parameter pin an edition. The IPC version of the index is in the file
names and in `text2ipc.json`; the package code vendored under `text2ipc/` is the one
the handler runs. Changes per version: `CHANGELOG.md` in the GitHub repository.

Contents: `scheme/` (titles and hierarchy), `index/` (vectors, parquet), `text2ipc/`
(the package, vendored), `text2ipc.json` (which index to serve), `handler.py`.

Data: IPC scheme master files by WIPO{data_note}. Embedder: `{embedder_model}` from the
Hub. Method, evaluation protocol and every measurement:
https://github.com/Jaimenms/text2ipc (`docs/methodology.md`, `docs/evals.md`).
A browser demo that needs no server runs at https://huggingface.co/spaces/jaimenms/text2ipc.
"""


def export_hf_repo(
    out: Path,
    *,
    version: str,
    lang: str = "EN",
    model: str,
    root: Path | None = None,
    package_src: Path | None = None,
    repo_id: str = "<user>/text2ipc-<lang>",
) -> Path:
    root = root or home()
    lang = lang.upper()
    from ..embeddings.base import model_slug

    ref = next(
        (
            r
            for r in available_indexes(root)
            if r.version == version and r.lang == lang and r.model_slug == model_slug(model)
        ),
        None,
    )
    if ref is None:
        raise FileNotFoundError(f"No index for {version} {lang} {model} in {root}")
    meta = IpcIndex.read_meta(ref.path)

    out.mkdir(parents=True, exist_ok=True)
    (out / "index").mkdir(exist_ok=True)
    (out / "scheme").mkdir(exist_ok=True)
    shutil.copy2(ref.path, out / "index" / ref.path.name)
    scheme = scheme_table_path(version, lang, root)
    shutil.copy2(scheme, out / "scheme" / scheme.name)

    src = package_src or Path(__file__).resolve().parents[1]
    dest = out / "text2ipc"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "hf"))
    (dest / "hf").mkdir()
    (dest / "hf" / "__init__.py").write_text("")

    shutil.copy2(Path(__file__).with_name("handler.py"), out / "handler.py")
    shutil.rmtree(out / "__pycache__", ignore_errors=True)  # left behind by local handler tests
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "text2ipc.json").write_text(
        json.dumps({"version": version, "lang": lang, "model": meta.model}, indent=2)
    )
    embedder_model = meta.model.split(":", 1)[1] if ":" in meta.model else meta.model
    (out / "README.md").write_text(
        MODEL_CARD.format(
            example_input=CARD_EXAMPLES.get(lang, CARD_EXAMPLES["EN"])[0],
            example_text=CARD_EXAMPLES.get(lang, CARD_EXAMPLES["EN"])[1],
            package_version=package_version,
            reranker_model=DEFAULT_RERANKER.partition(":")[2],
            version=version,
            lang=lang,
            lang_tag=lang.lower(),
            repo_id=repo_id,
            embedder_model=embedder_model,
            data_note=(
                "; Portuguese titles from INPI Brazil's IPC Publication"
                if meta.titles_overlay
                else ""
            ),
        )
    )
    (out / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    return out
