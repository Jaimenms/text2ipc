"""Discover published IPC versions.

WIPO publishes one directory per version named ``YYYYMMDD`` under the IT support area.
The next year's version is usually published months before it enters into force, so
``latest`` (newest published) and ``current`` (newest in force today) can differ.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import httpx

from .config import WIPO_BASE_URL, home

_VERSION_RE = re.compile(r"/(\d{8})/")


def parse_version_index(html: str) -> list[str]:
    """Extract ``YYYYMMDD`` directory names from the WIPO index page, ascending."""
    found = {m.group(1) for m in _VERSION_RE.finditer(html)}
    return sorted(v for v in found if _is_date(v))


def _is_date(v: str) -> bool:
    try:
        dt.date(int(v[:4]), int(v[4:6]), int(v[6:]))
    except ValueError:
        return False
    return True


def list_remote_versions(timeout: float = 30.0) -> list[str]:
    """Versions published on wipo.int, ascending. Requires network."""
    resp = httpx.get(f"{WIPO_BASE_URL}/", timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return parse_version_index(resp.text)


def list_local_versions(root: Path | None = None) -> list[str]:
    """Versions for which a scheme XML has already been downloaded, ascending."""
    wipo_dir = (root or home()) / "wipo"
    if not wipo_dir.exists():
        return []
    found = {
        m.group(1)
        for p in wipo_dir.glob("*_ipc_scheme_*.xml")
        if (m := re.search(r"_ipc_scheme_(\d{8})\.xml$", p.name))
    }
    return sorted(found)


def resolve_version(
    spec: str = "latest",
    *,
    today: dt.date | None = None,
    offline: bool = False,
    root: Path | None = None,
) -> str:
    """Turn ``latest`` / ``current`` / ``YYYYMMDD`` into a concrete version string.

    ``latest`` is the newest version WIPO has published, even if not yet in force.
    ``current`` is the newest version whose date is not after ``today``.
    """
    if re.fullmatch(r"\d{8}", spec):
        return spec
    if spec not in {"latest", "current"}:
        raise ValueError(f"Unknown version spec {spec!r}; use 'latest', 'current' or YYYYMMDD")
    versions = list_local_versions(root) if offline else list_remote_versions()
    if not offline:
        versions = sorted(set(versions) | set(list_local_versions(root)))
    if not versions:
        raise RuntimeError("No IPC versions found (offline and nothing downloaded yet)")
    if spec == "latest":
        return versions[-1]
    today = today or dt.date.today()
    in_force = [v for v in versions if dt.date(int(v[:4]), int(v[4:6]), int(v[6:])) <= today]
    if not in_force:
        raise RuntimeError(f"No IPC version in force on {today}")
    return in_force[-1]
