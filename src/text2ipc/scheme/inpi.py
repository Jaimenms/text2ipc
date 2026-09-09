"""Portuguese IPC titles from INPI Brazil's IPC Publication (IPCPUB) instance.

WIPO publishes authentic master files in English and French only. For other languages
its IPC Publication "bridges" to national offices that run the same IPCPUB software
with a translation loaded; for Portuguese that is INPI Brazil at
``https://ipc.inpi.gov.br/classifications/ipc/ipcpub/``.

That instance serves the translated scheme as static JSON, one file per subclass plus
an index with sections, classes and subclasses::

    <base>/media/<version>/<timestamp>/IPC/scheme/pt/json/index.json
    <base>/media/<version>/<timestamp>/IPC/scheme/pt/json/<SUBCLASS>.json

The ``timestamp`` per version is embedded in the IPCPUB HTML page. Titles are HTML:
cross references sit in ``<span>`` and the edition tag in ``<a rel="versions">``; both
are stripped so the result matches what :mod:`parse` extracts from the WIPO XML.

Every JSON file is cached under ``<home>/inpi/<version>/<lang>/``; that cache is the
source of truth and titles are re-derived from it on each build (a few seconds).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..config import home
from .symbols import normalize_symbol

INPI_BASE = "https://ipc.inpi.gov.br/classifications/ipc/ipcpub"
_HEADERS = {"User-Agent": "Mozilla/5.0 (text2ipc)"}
_TIMESTAMPS_RE = re.compile(r"timestamps\s*:\s*\{([^}]*)\}")
_PAIR_RE = re.compile(r'"(\d{8})"\s*:\s*"(\d{14})"')
_SECTION_PREFIX_RE = re.compile(r"^\s*SE[ÇC][ÃA]O\s+[A-H]\s*[—–-]\s*", re.IGNORECASE)


def fetch_inpi_titles(
    version: str,
    *,
    lang: str = "pt",
    root: Path | None = None,
    force: bool = False,
    pause: float = 0.2,
    timeout: float = 60.0,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Return ``{canonical symbol: title}`` for every entry, from the JSON cache or INPI."""
    cache = (root or home()) / "inpi" / version / lang.lower()
    if force and cache.exists():
        for f in cache.glob("*.json"):
            f.unlink()
    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        stamp = _timestamp_for(client, version)
        base = f"{INPI_BASE}/media/{version}/{stamp}/IPC/scheme/{lang}/json"
        index = _get_json(client, f"{base}/index.json", cache / "index.json")
        titles: dict[str, str] = {}
        subclasses: list[str] = []
        for entry in _walk(index):
            if entry.get("kind") in {"s", "c", "u"} and entry.get("symbol"):
                symbol = _symbol(entry["symbol"])
                titles[symbol] = clean_title(entry.get("title1") or "", entry["kind"])
                if entry["kind"] == "u":
                    subclasses.append(symbol)
        for i, subclass in enumerate(subclasses, 1):
            cached = (cache / f"{subclass}.json").exists()
            data = _get_json(client, f"{base}/{subclass}.json", cache / f"{subclass}.json")
            for entry in _walk(data):
                kind = entry.get("kind", "")
                if (kind == "m" or kind.isdigit()) and entry.get("symbol"):
                    titles[_symbol(entry["symbol"])] = clean_title(entry.get("title1") or "", kind)
            if progress:
                progress(i, len(subclasses))
            if not cached:
                time.sleep(pause)
    return titles


def clean_title(html: str, kind: str = "") -> str:
    """Strip cross references, edition tags and markup from an IPCPUB title."""
    html = re.sub(r"<a rel=\"versions\".*?</a>", " ", html, flags=re.DOTALL)
    html = re.sub(r"<strong>\s*\[\d{4}\.\d{2}\]\s*</strong>", " ", html)
    html = re.sub(r"<span>.*?</span>", " ", html, flags=re.DOTALL)
    if (m := re.search(r'<div class="txt[^"]*">(.*)$', html, flags=re.DOTALL)) is not None:
        html = m.group(1)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\(\s*\)", " ", text)  # empty parentheses left by removed refs
    text = re.sub(r"\[\d{4}\.\d{2}\]", " ", text)  # edition tags printed without a link
    text = " ".join(text.replace("\xa0", " ").split())
    if kind == "s":
        text = _SECTION_PREFIX_RE.sub("", text)
    return text.strip(" ;")


def _timestamp_for(client: httpx.Client, version: str) -> str:
    page = client.get(f"{INPI_BASE}/", params={"notion": "scheme", "version": version})
    page.raise_for_status()
    m = _TIMESTAMPS_RE.search(page.text)
    stamps = dict(_PAIR_RE.findall(m.group(1))) if m else {}
    if version not in stamps:
        raise RuntimeError(
            f"INPI IPCPUB has no Portuguese data for IPC {version}; available: {sorted(stamps)}"
        )
    return stamps[version]


def _get_json(client: httpx.Client, url: str, cache_file: Path, attempts: int = 10):
    """Fetch with retries and exponential backoff; INPI answers 503 now and then.

    Every file is cached raw on disk, so an interrupted run resumes where it stopped.
    """
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as e:
            last = e
            time.sleep(min(2.0**attempt, 60.0))
    raise RuntimeError(f"Could not fetch {url}") from last


def _walk(items):
    for item in items or []:
        yield item
        yield from _walk(item.get("children"))


def _symbol(raw: str) -> str:
    return normalize_symbol(raw.replace("\xa0", " "))
