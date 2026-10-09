"""Assemble a static Hugging Face Space that classifies in the visitor's browser.

Static Spaces are free; Spaces that run Python are not. So the demo ships the index
as small static files and does everything client side: ``transformers.js`` embeds
the query with the ONNX twin of the embedder, and ``scorer.js`` (a port of
``search/scorer.py``) ranks the entries. See ADR 0007.

Layout written by :func:`export_web_demo`::

    index.html, app.js, scorer.js   the page (copied from ``web/static/``)
    manifest.json                   embedder, query prefix, one entry per index
    data/<lang>/scheme.json         columns symbol, level, depth, parent, title, heading
    data/<lang>/vectors.bin         per-row float32 scales, then int8 vectors (or float32)
    README.md                       Space front matter (sdk: static) and a description
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .. import __version__ as package_version
from ..config import LEVELS, home
from ..embeddings.st import prefixes_for
from ..index import IpcIndex, available_indexes, scheme_table_path
from ..index.paths import IndexRef

#: The package default. 279 MB quantised in the browser; e5-small is 118 MB but ranks
#: generic entries (fire hoses) near the top of unrelated texts (ADR 0007). Any model
#: with an ONNX twin on the Hub works.
WEB_DEFAULT_MODEL = "st:intfloat/multilingual-e5-base"

#: sentence-transformers model -> transformers.js model (ONNX weights on the Hub).
WEB_MODELS = {
    "intfloat/multilingual-e5-small": "Xenova/multilingual-e5-small",
    "intfloat/multilingual-e5-base": "Xenova/multilingual-e5-base",
    "intfloat/multilingual-e5-large": "Xenova/multilingual-e5-large",
}

STATIC_FILES = ("index.html", "app.js", "scorer.js", "rerank-worker.js")

#: Token limit the browser applies before chunking a long text (sentence-transformers
#: ``max_seq_length`` of the model); e5 models take 512.
WEB_MAX_TOKENS = {"intfloat/multilingual-e5-small": 512, "intfloat/multilingual-e5-base": 512}

#: Cross-encoder the page offers as an opt-in second stage (ONNX on the Hub). jina v2
#: in 8 bits equals bge-reranker-v2-m3 at @1 on the evals at half the size (docs/evals.md).
WEB_DEFAULT_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
WEB_RERANKER_DTYPE = "q8"
WEB_RERANK_CANDIDATES = 25
WEB_RERANK_MAX_LENGTH = 768

SPACE_README = """---
title: text2ipc
emoji: 🏷️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Patent text to IPC symbols, computed in your browser
models:
{models}
---

# text2ipc, in the browser

Paste a patent abstract (Portuguese, English or any language the embedder knows) and
get a ranked list of International Patent Classification symbols. Nothing is sent to
a server: the page downloads the quantised embedder `{web_model}` ({web_dtype}) and the
IPC index once, then embeds the query and scores it against the hierarchy locally.

Indexes in this Space: {index_list}. The "Rerank" option loads a second model,
`{reranker}` in 8 bits, that re-judges the best candidates against the text
(that model is released under CC BY-NC 4.0: this demo is non-commercial; the
package's default reranker, `BAAI/bge-reranker-v2-m3`, is Apache-2.0).

How it works, the evaluation numbers and the Python package are at
https://github.com/Jaimenms/text2ipc. The same index runs locally with
`t2ipc classify` and as a Hugging Face Inference Endpoint.

Data: IPC scheme master files by WIPO{data_note}. Scoring: `scorer.js` is a port of
`text2ipc/search/scorer.py`; vectors are stored as int8 with a per-row scale.
"""


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8: ``v ~= q * scale`` with ``scale = max|v| / 127``."""
    v = np.asarray(vectors, dtype=np.float32)
    scales = np.max(np.abs(v), axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(v / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def dequantize_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


def write_vectors(path: Path, vectors: np.ndarray, encoding: str = "int8") -> dict:
    """Write ``vectors.bin``: for int8, ``rows`` little-endian float32 scales followed by
    the ``rows x dim`` int8 matrix; for float32, the matrix alone."""
    rows, dim = vectors.shape
    if encoding == "int8":
        q, scales = quantize_int8(vectors)
        payload = scales.astype("<f4").tobytes() + np.ascontiguousarray(q).tobytes()
    elif encoding == "float32":
        payload = np.ascontiguousarray(vectors, dtype="<f4").tobytes()
    else:
        raise ValueError(f"encoding must be int8 or float32, got {encoding!r}")
    path.write_bytes(payload)
    return {"rows": rows, "dim": dim, "encoding": encoding, "bytes": len(payload)}


def scheme_columns(index: IpcIndex) -> dict[str, list]:
    """The scheme table as parallel columns, rows in index order, parent as a row number."""
    return {
        "symbol": index.symbols,
        "level": [int(x) for x in index.levels],
        "depth": [n.depth for n in index.nodes],
        "parent": [int(p) for p in index.parent_idx],
        "title": [n.title for n in index.nodes],
        "heading": [n.heading for n in index.nodes],
    }


def web_model_for(model: str) -> str | None:
    backend, _, name = model.partition(":")
    if backend != "st":
        return None
    return WEB_MODELS.get(name, name)


def export_web_demo(
    out: Path,
    *,
    model: str = WEB_DEFAULT_MODEL,
    langs: tuple[str, ...] | list[str] = ("PT",),
    version: str = "latest",
    root: Path | None = None,
    repo_id: str = "<user>/text2ipc",
    web_model: str | None = None,
    web_dtype: str = "q8",
    encoding: str = "int8",
    examples: Path | None = None,
    web_reranker: str | None = WEB_DEFAULT_RERANKER,
) -> Path:
    """Write the static Space into ``out``; one index per language, all on ``model``.

    ``version`` is resolved per language against the indexes built for ``model``.
    ``examples`` is a JSONL of eval cases (``t2ipc rpi`` format); only their title and
    abstract are shipped, as sample texts the page offers to fill the box with.
    """
    from ..classifier import resolve_built_version
    from ..embeddings.base import model_slug

    root = root or home()
    web_model = web_model or web_model_for(model)
    if web_model is None:
        raise ValueError(f"No browser model known for {model!r}; pass web_model explicitly")
    query_prefix, _ = prefixes_for(model.partition(":")[2])

    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    entries = []
    dim: int | None = None
    data_notes = []
    for lang in langs:
        lang = lang.upper()
        built = resolve_built_version(version, lang, model, root)
        ref = _find(root, built, lang, model_slug(model))
        index = IpcIndex.read(ref.path, scheme_table_path(built, lang, root))
        if dim is not None and index.meta.dim != dim:
            raise ValueError("All indexes in one Space must share the embedder")
        dim = index.meta.dim
        d = out / "data" / lang.lower()
        d.mkdir(exist_ok=True)
        (d / "scheme.json").write_text(json.dumps(scheme_columns(index), ensure_ascii=False))
        vec = write_vectors(d / "vectors.bin", index.vectors, encoding)
        entries.append(
            {
                "lang": lang,
                "version": built,
                "rows": vec["rows"],
                "titles_overlay": index.meta.titles_overlay,
                "scheme": f"data/{lang.lower()}/scheme.json",
                "vectors": f"data/{lang.lower()}/vectors.bin",
                "encoding": encoding,
                "bytes": vec["bytes"],
            }
        )
        if index.meta.titles_overlay:
            data_notes.append(lang)

    manifest = {
        "generator": f"text2ipc {package_version}",
        "model": model,
        "web_model": web_model,
        "web_dtype": web_dtype,
        "query_prefix": query_prefix,
        "max_tokens": WEB_MAX_TOKENS.get(model.partition(":")[2], 512),
        "reranker": (
            {
                "web_model": web_reranker,
                "dtype": WEB_RERANKER_DTYPE,
                "candidates": WEB_RERANK_CANDIDATES,
                "max_length": WEB_RERANK_MAX_LENGTH,
                "fusion": "blend",
            }
            if web_reranker
            else None
        ),
        "dim": dim,
        "levels": list(LEVELS),
        "indexes": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    static = Path(__file__).with_name("static")
    for name in STATIC_FILES:
        shutil.copy2(static / name, out / name)
    if examples is not None:
        (out / "examples.json").write_text(
            json.dumps(load_examples(examples), ensure_ascii=False, indent=1)
        )
    (out / "package.json").write_text(json.dumps({"type": "module", "private": True}) + "\n")
    (out / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    (out / "README.md").write_text(
        SPACE_README.format(
            models="\n".join(
                f"  - {m}" for m in [web_model, *([web_reranker] if web_reranker else [])]
            ),
            web_model=web_model,
            web_dtype=web_dtype,
            index_list=", ".join(f"IPC {e['version']} {e['lang']}" for e in entries),
            reranker=web_reranker or "none",
            data_note=(
                "; Portuguese titles from INPI Brazil's IPC Publication"
                if "PT" in data_notes
                else ""
            ),
        )
    )
    return out


def load_examples(path: Path) -> list[dict]:
    """Language, title and abstract of each case; nothing else reaches the page. The page
    shows the examples whose language matches the selected scheme."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if not (c.get("title") and c.get("abstract")):
            raise ValueError(f"example {c.get('id')} needs both a title and an abstract")
        out.append(
            {
                "lang": (c.get("lang") or "pt").upper(),
                "title": c["title"],
                "abstract": c["abstract"],
            }
        )
    return out


def _find(root: Path, version: str, lang: str, slug: str) -> IndexRef:
    for r in available_indexes(root):
        if r.version == version and r.lang == lang and r.model_slug == slug:
            return r
    raise FileNotFoundError(f"No index for {version} {lang} {slug} in {root}")
