"""Assemble a Hugging Face model repository directory for Inference Endpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import home
from ..index import IpcIndex
from ..index.paths import available_indexes, scheme_table_path

REQUIREMENTS = """numpy>=1.26
httpx>=0.27
typer>=0.12
rich>=13.7
sentence-transformers>=3.0
"""

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
  - text2ipc
base_model: {embedder_model}
---

# text2ipc: IPC {version} ({lang}) via {embedder_model}

Maps free text (a patent abstract, for example) to ranked International Patent
Classification symbols. Every IPC entry is embedded from its full ancestor path and the
query is scored against the hierarchy (path support, beam by best subtree score).

This repository is a **custom Inference Endpoints handler** (`handler.py`). Deploy it as
an Inference Endpoint and call:

```json
{{"inputs": "enxada manual com duas lâminas para capina",
 "parameters": {{"level": "group", "top_k": 5}}}}
```

`level` is one of section, class, subclass, group, subgroup or auto. The response is a
list of `{{symbol, canonical, level, score, similarity, title, path}}`.

## Use it locally

```bash
pip install "text2ipc[st] @ git+https://github.com/jaimenms/study-text-to-ipc"
t2ipc download {repo_id}
t2ipc classify "Aparelho para combate a incêndios com mangueira flexível" --lang {lang} --level group
```

```python
from text2ipc import IpcClassifier
clf = IpcClassifier("{version}", lang="{lang}")
for m in clf.classify("...", level="group", top_k=5):
    print(m.pretty, round(m.score, 3), m.text)
```

Contents: `scheme/` (titles and hierarchy), `index/` (vectors, parquet), `text2ipc/`
(the package, vendored), `text2ipc.json` (which index to serve).

Data: IPC scheme master files by WIPO{data_note}. Embedder: `{embedder_model}` from the Hub.
Accuracy on INPI-published applications is documented in the text2ipc repository.
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
