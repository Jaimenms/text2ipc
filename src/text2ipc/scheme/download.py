"""Fetch and cache WIPO IPC scheme master files."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx

from ..config import WIPO_BASE_URL, WIPO_LANGS, home


def scheme_url(version: str) -> str:
    return f"{WIPO_BASE_URL}/{version}/MasterFiles/ipc_scheme_{version}.zip"


def scheme_path(version: str, lang: str = "EN", root: Path | None = None) -> Path:
    """Raw WIPO master file, kept under ``wipo/``; derived tables go to ``scheme/``."""
    return (root or home()) / "wipo" / f"{lang.upper()}_ipc_scheme_{version}.xml"


def fetch_scheme(
    version: str,
    lang: str = "EN",
    *,
    root: Path | None = None,
    force: bool = False,
    timeout: float = 300.0,
) -> Path:
    """Return the local path of ``<LANG>_ipc_scheme_<version>.xml``, downloading if needed.

    The zip holds every language WIPO ships, so one download serves all of them.
    """
    lang = lang.upper()
    if lang not in WIPO_LANGS:
        raise ValueError(f"WIPO master files are only published in {WIPO_LANGS}, not {lang}")
    target = scheme_path(version, lang, root)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(scheme_url(version), timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".xml"):
                (target.parent / Path(name).name).write_bytes(zf.read(name))
    if not target.exists():
        raise FileNotFoundError(f"{target.name} not found in {scheme_url(version)}")
    return target
