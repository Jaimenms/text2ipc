"""Eval cases from the *Revista da Propriedade Industrial* (RPI), INPI Brazil.

Every week INPI publishes a zip at ``https://revistas.inpi.gov.br/txt/P<issue>.zip``.
Recent issues hold ``Patente_<issue>_<date>.xml``; older ones (e.g. 2100, from 2011)
hold only ``P<issue>.txt``, a Latin-1 text file with one ``(INID) value`` field per
line and a ``(Cd) <code>`` line opening every dispatch. Both formats are parsed.

Dispatch codes 1.3 (PCT application admitted to the national phase) and 3.1
(application published) expose a title (INID 54), the IPC symbols (INID 51) and, in
issues that include it, the abstract (INID 57). Recent XML issues carry no abstract;
the text issues around 2100 do, which makes them the better eval source. Text is
Portuguese.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import httpx

from ..config import home
from ..scheme.symbols import normalize_symbol
from .cases import EvalCase

RPI_URL = "https://revistas.inpi.gov.br/txt/P{issue}.zip"
PUBLISHED_DISPATCHES = ("1.3", "3.1")


def fetch_rpi(issue: int | str, *, root: Path | None = None, timeout: float = 120.0) -> Path:
    """Download the issue zip (cached under ``<home>/rpi``) and return the XML path."""
    target_dir = (root or home()) / "rpi"
    existing = sorted(target_dir.glob(f"Patente_{issue}_*.xml")) or sorted(
        target_dir.glob(f"P{issue}.txt")
    )
    if existing:
        return existing[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    content = _download(RPI_URL.format(issue=issue), timeout=timeout)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")] or [
            n for n in zf.namelist() if n.lower().endswith(".txt")
        ]
        if not names:
            raise FileNotFoundError(f"No XML or TXT inside {RPI_URL.format(issue=issue)}")
        out = target_dir / Path(names[0]).name
        out.write_bytes(zf.read(names[0]))
    return out


def _download(url: str, *, timeout: float, attempts: int = 3) -> bytes:
    """INPI's server sometimes closes the connection early; retry a few times."""
    headers = {"User-Agent": "Mozilla/5.0 (text2ipc)", "Accept-Encoding": "identity"}
    last: Exception | None = None
    for _ in range(attempts):
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers=headers)
            resp.raise_for_status()
            return resp.content
        except (httpx.TransportError, httpx.HTTPStatusError) as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"Could not download {url} after {attempts} attempts") from last


def parse_rpi(path: Path | str, dispatches: tuple[str, ...] = PUBLISHED_DISPATCHES) -> list[dict]:
    """Raw records: issue, number, dispatch, title, abstract, ipc list (as printed)."""
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return parse_rpi_text(path, dispatches)
    return parse_rpi_xml(path, dispatches)


_FIELD_RE = re.compile(r"^\((?P<inid>[A-Za-z0-9]{2})\)\s?(?P<value>.*)$")
_IPC_RE = re.compile(r"[A-H]\d{2}[A-Z]\s?\d{1,4}/\d{2,6}")


def parse_rpi_text(
    path: Path | str, dispatches: tuple[str, ...] = PUBLISHED_DISPATCHES
) -> list[dict]:
    """Parse the legacy ``P<issue>.txt`` layout."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    header = re.search(r"^\s*N[o°º]\s*(\d+)\s+de\s", text, flags=re.MULTILINE)
    fname = re.search(r"P(\d+)", path.stem)
    issue_no = header.group(1) if header else (fname.group(1) if fname else "")
    records: list[dict] = []
    current: dict | None = None
    last_field: str | None = None
    for line in text.splitlines():
        m = _FIELD_RE.match(line)
        if m is None:
            if current is not None and last_field and line.strip():
                current[last_field] += " " + line.strip()
            continue
        inid, value = m.group("inid"), m.group("value").strip()
        if inid == "Cd":
            if current is not None:
                records.append(current)
            current = {"dispatch": value, "21": "", "51": "", "54": "", "57": ""}
            last_field = None
            continue
        if current is None:
            continue
        if inid in current and inid != "dispatch":
            current[inid] = (current[inid] + " " + value).strip() if current[inid] else value
            last_field = inid
        else:
            last_field = None
    if current is not None:
        records.append(current)

    out = []
    for r in records:
        if r["dispatch"] not in dispatches:
            continue
        title = _clean(r["54"])
        abstract = _strip_repeated_title(_clean(r["57"]), title)
        out.append(
            {
                "issue": issue_no,
                "number": _clean(r["21"]),
                "dispatch": r["dispatch"],
                "title": title,
                "abstract": abstract,
                "ipc": [m.group(0) for m in _IPC_RE.finditer(r["51"])],
            }
        )
    return out


def _strip_repeated_title(abstract: str, title: str) -> str:
    """Text issues print the title again at the start of the abstract, with slightly
    different spacing and punctuation, so compare on letters and digits only."""
    key = _alnum(title)
    if not key or not _alnum(abstract).startswith(key):
        return abstract
    consumed = 0
    for i, ch in enumerate(abstract):
        if ch.isalnum():
            consumed += 1
            if consumed == len(key):
                return abstract[i + 1 :].lstrip(" .:;,-")
    return abstract


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def parse_rpi_xml(
    xml_path: Path | str, dispatches: tuple[str, ...] = PUBLISHED_DISPATCHES
) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    issue = root.get("numero", "")
    records = []
    for d in root.iter("despacho"):
        code = (d.findtext("codigo") or "").strip()
        if code not in dispatches:
            continue
        pp = d.find("processo-patente")
        if pp is None:
            continue
        number = (pp.findtext("numero") or "").strip()
        title = _clean(pp.findtext("titulo"))
        abstract = _clean(_first_text(pp, ("resumo", ".//*[@inid='57']")))
        ipc = [
            (e.text or "").strip()
            for e in pp.iter("classificacao-internacional")
            if (e.text or "").strip()
        ]
        records.append(
            {
                "issue": issue,
                "number": number,
                "dispatch": code,
                "title": title,
                "abstract": abstract,
                "ipc": ipc,
            }
        )
    return records


def cases_from_rpi(records: list[dict], *, require_abstract: bool = False) -> list[EvalCase]:
    cases = []
    for r in records:
        text = r["abstract"] or r["title"]
        if not text or not r["ipc"] or (require_abstract and not r["abstract"]):
            continue
        symbols = []
        for raw in r["ipc"]:
            try:
                symbols.append(normalize_symbol(raw))
            except ValueError:
                continue
        if not symbols:
            continue
        cases.append(
            EvalCase(
                id=f"rpi{r['issue']}:{r['number'].replace(' ', '')}",
                text=text,
                ipc=tuple(dict.fromkeys(symbols)),
                lang="pt",
                title=r["title"] or None,
                abstract=r["abstract"] or None,
                source=f"RPI {r['issue']} despacho {r['dispatch']}",
                tags=("abstract" if r["abstract"] else "title-only", f"despacho-{r['dispatch']}"),
            )
        )
    return cases


def _first_text(el: ET.Element, paths: tuple[str, ...]) -> str | None:
    for p in paths:
        found = el.find(p)
        if found is not None and (found.text or "").strip():
            return found.text
    return None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
