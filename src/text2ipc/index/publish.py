"""Fetch a prebuilt index from a Hugging Face model repository made by ``t2ipc hf-export``.

Building an index takes minutes and a model download, so end users fetch the two
Parquet tables (``index/`` and ``scheme/``) from the Hub instead. The repository's
``text2ipc.json`` says which version, language and model they belong to.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import home


def download_from_hf(
    repo_id: str, *, root: Path | None = None, revision: str | None = None, token=None
) -> tuple[Path, dict]:
    """Copy ``index/*.parquet`` and ``scheme/*.parquet`` from a Hub model repo into home.

    Returns the index path and the repo's ``text2ipc.json`` (version, lang, model).
    """
    from huggingface_hub import HfApi, hf_hub_download

    root = root or home()
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="model", revision=revision)
    wanted = [f for f in files if f.endswith(".parquet") and f.split("/")[0] in {"index", "scheme"}]
    if "text2ipc.json" not in files or not wanted:
        raise FileNotFoundError(f"{repo_id}: no text2ipc.json or parquet tables found")
    cfg_path = hf_hub_download(repo_id, "text2ipc.json", revision=revision, token=token)
    cfg = json.loads(Path(cfg_path).read_text())
    index_target: Path | None = None
    for f in wanted:
        cached = Path(hf_hub_download(repo_id, f, revision=revision, token=token))
        target = root / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        if f.startswith("index/"):
            index_target = target
    assert index_target is not None
    return index_target, cfg
