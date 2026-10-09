"""Package-wide constants and locations."""

from __future__ import annotations

import os
from pathlib import Path

#: Hierarchy levels, top to bottom. ``depth`` further refines subgroups by dot count.
LEVELS: tuple[str, ...] = ("section", "class", "subclass", "group", "subgroup")

#: Special level meaning "descend while the evidence supports it".
AUTO_LEVEL = "auto"

#: Scheme languages WIPO ships in its master files.
WIPO_LANGS: tuple[str, ...] = ("EN", "FR")
#: Languages available as a title overlay on the English hierarchy, and their source.
OVERLAY_LANGS: dict[str, str] = {"PT": "inpi"}
DEFAULT_LANG = "EN"

#: Multilingual by default so that a Portuguese abstract can be matched against the
#: English scheme. ``e5-base`` roughly doubles precision over ``e5-small`` on the RPI
#: evals (docs/evals.md). Override with ``TEXT2IPC_MODEL`` or the ``model=`` argument.
DEFAULT_MODEL = "st:intfloat/multilingual-e5-base"

#: Cross-encoder that re-scores the first stage's candidates when ``rerank`` is on.
#: Multilingual, 568M parameters; lifts subclass@1 by 9 points on the PT evals.
DEFAULT_RERANKER = "ce:BAAI/bge-reranker-v2-m3"

WIPO_BASE_URL = "https://www.wipo.int/ipc/itos4ipc/ITSupport_and_download_area"


#: Hugging Face model repository holding the published Portuguese index.
DEFAULT_HF_REPO = "jaimenms/text2ipc-pt"


def home() -> Path:
    """Directory holding downloaded schemes and built indexes.

    Resolution order:

    1. ``TEXT2IPC_HOME``.
    2. ``<repo>/data`` when the current directory (or one of its parents) is the
       text2ipc study repository and that directory exists, so artefacts stay next to
       the code while developing, also from ``notebooks/`` or ``tests/``.
    3. ``$XDG_CACHE_HOME/text2ipc`` or ``~/.cache/text2ipc`` for installed users.
    """
    if env := os.environ.get("TEXT2IPC_HOME"):
        return Path(env).expanduser()
    if (repo_data := _repo_data_dir()) is not None:
        return repo_data
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "text2ipc"


def _repo_data_dir(cwd: Path | None = None) -> Path | None:
    for candidate in [cwd or Path.cwd(), *(cwd or Path.cwd()).parents]:
        pyproject = candidate / "pyproject.toml"
        try:
            is_repo = pyproject.is_file() and 'name = "text2ipc"' in pyproject.read_text()
        except OSError:
            return None
        if is_repo:
            data = candidate / "data"
            return data if data.is_dir() else None
    return None


def default_model() -> str:
    return os.environ.get("TEXT2IPC_MODEL", DEFAULT_MODEL)
