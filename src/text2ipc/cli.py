"""``t2ipc`` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import (
    DEFAULT_HF_REPO,
    DEFAULT_RERANKER,
    LEVELS,
    OVERLAY_LANGS,
    WIPO_LANGS,
    default_model,
    home,
)
from .embeddings import get_embedder
from .index import (
    IpcIndex,
    available_indexes,
    build_index,
    find_previous_index,
    index_path,
    scheme_table_path,
)
from .scheme import (
    SchemeTable,
    apply_titles,
    fetch_scheme,
    format_symbol,
    load_titles_csv,
    parse_scheme,
)
from .versions import list_local_versions, list_remote_versions, resolve_version
from .web.export import WEB_DEFAULT_MODEL, WEB_DEFAULT_RERANKER

app = typer.Typer(help="Map free text to IPC symbols.", no_args_is_help=True)
console = Console()


@app.command()
def versions(remote: bool = typer.Option(True, help="Query wipo.int; --no-remote lists local")):
    """List IPC versions, newest last."""
    for v in list_remote_versions() if remote else list_local_versions():
        console.print(v)


@app.command()
def indexes():
    """List indexes built under the text2ipc home."""
    table = Table("version", "lang", "model", "rows", "MB", "path")
    for ref in available_indexes():
        meta = IpcIndex.read_meta(ref.path)
        table.add_row(
            ref.version,
            ref.lang,
            meta.model,
            str(meta.rows),
            f"{ref.path.stat().st_size / 1e6:.0f}",
            str(ref.path),
        )
    console.print(table)
    console.print(f"home: {home()}")


@app.command()
def build(
    version: str = typer.Option(
        "latest", help="YYYYMMDD, 'latest' (newest published) or 'current' (in force)"
    ),
    lang: str = typer.Option(
        "EN", help="Scheme language: EN or FR (WIPO master files), PT (INPI translation)"
    ),
    model: str = typer.Option(None, help="Embedder spec, e.g. st:intfloat/multilingual-e5-base"),
    previous: str = typer.Option(
        "auto", help="'auto': newest older index for the same lang+model; 'none'; or a path"
    ),
    titles_csv: Path = typer.Option(None, help="symbol,title CSV overriding entry titles"),
    out: Path = typer.Option(None, help="Output parquet; default is under the text2ipc home"),
    batch_size: int = typer.Option(256),
    force_download: bool = typer.Option(False, help="Re-download the WIPO zip / INPI titles"),
):
    """Download the WIPO scheme for a version, write the scheme table and the index."""
    model = model or default_model()
    version = resolve_version(version)
    lang = lang.upper()
    scheme, overlay = _prepare_scheme(version, lang, titles_csv, force_download)

    prev: tuple[Path, Path] | None = None
    found = None
    if previous == "auto":
        found = find_previous_index(version, lang, model)
    elif previous != "none":
        found = Path(previous)
    if found is not None:
        prev = (found, scheme_table_path(IpcIndex.read_meta(found).version, lang))
        console.print(f"reusing vectors from {found.name}")

    embedder = get_embedder(model)
    with console.status("embedding...") as status:
        index, report = build_index(
            scheme,
            embedder,
            version=version,
            lang=lang,
            previous=prev,
            titles_overlay=overlay,
            batch_size=batch_size,
            progress=lambda done, total: status.update(f"embedding {done}/{total}"),
        )
    target = out or index_path(version, lang, model)
    index.write(target)
    console.print(report.summary())
    console.print(f"wrote {target}")


def _prepare_scheme(
    version: str, lang: str, titles_csv: Path | None, force: bool
) -> tuple[SchemeTable, str | None]:
    """Parse the WIPO XML (plus a title overlay for PT), write and return the scheme table."""
    if lang in WIPO_LANGS:
        xml_path = fetch_scheme(version, lang, force=force)
    elif lang in OVERLAY_LANGS:
        xml_path = fetch_scheme(version, "EN", force=force)
    else:
        raise typer.BadParameter(f"lang must be one of {WIPO_LANGS + tuple(OVERLAY_LANGS)}")
    nodes = parse_scheme(xml_path)
    overlay = None
    if titles_csv is not None:
        nodes = apply_titles(nodes, load_titles_csv(titles_csv))
        overlay = str(titles_csv)
    elif lang in OVERLAY_LANGS:
        from .scheme import fetch_inpi_titles

        with console.status("fetching Portuguese titles from INPI...") as status:
            titles = fetch_inpi_titles(
                version,
                force=force,
                progress=lambda i, n: status.update(f"INPI titles: subclass {i}/{n}"),
            )
        nodes = apply_titles(nodes, titles)
        overlay = f"{OVERLAY_LANGS[lang]}:{lang.lower()}"
        console.print(f"{len(titles)} Portuguese titles applied")
    scheme = SchemeTable.from_nodes(nodes)
    target = scheme_table_path(version, lang)
    scheme.write(target)
    console.print(
        f"[bold]{version} {lang}[/]: {len(scheme)} entries from {xml_path.name} -> {target.name}"
    )
    return scheme, overlay


@app.command("scheme")
def scheme_cmd(
    version: str = typer.Option("latest", help="YYYYMMDD, 'latest' or 'current'"),
    lang: str = typer.Option("EN", help="EN, FR (WIPO) or PT (INPI titles)"),
    titles_csv: Path = typer.Option(None, help="symbol,title CSV overriding entry titles"),
    force_download: bool = typer.Option(False),
):
    """Write only the scheme table (titles + hierarchy) for a version and language."""
    version = resolve_version(version)
    _prepare_scheme(version, lang.upper(), titles_csv, force_download)


@app.command()
def classify(
    text: str = typer.Argument(..., help="Abstract or any free text; use '-' to read stdin"),
    version: str = typer.Option(
        "latest", help="YYYYMMDD, 'latest' or 'current' among built indexes"
    ),
    level: str = typer.Option("subgroup", help=f"{', '.join(LEVELS)} or auto"),
    top_k: int = typer.Option(10),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    gap: float = typer.Option(None, help="Drop results more than this below the best score"),
    chunking: str = typer.Option(
        "mean", help="Long texts: 'mean' of chunk vectors, 'max' over chunks, or 'truncate'"
    ),
    rerank: bool = typer.Option(False, help="Second stage: cross-encoder over the candidates"),
    reranker: str = typer.Option(None, help=f"Reranker spec (default {DEFAULT_RERANKER})"),
    candidates: int = typer.Option(None, help="First-stage list the reranker judges"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Rank IPC symbols for a text of any length (a whole description is chunked)."""
    from .classifier import IpcClassifier

    if text == "-":
        text = sys.stdin.read()
    clf = IpcClassifier(version, lang=lang, model=model)
    matches = clf.classify(
        text,
        level=level,
        top_k=top_k,
        gap=gap,
        chunking=chunking,
        rerank=reranker or rerank,
        candidates=candidates,
    )
    if as_json:
        print(json.dumps([m.__dict__ | {"pretty": m.pretty} for m in matches], indent=2))
        return
    reranked = rerank or reranker
    columns = (
        "#",
        "symbol",
        "score",
        "sim",
        *(("judge",) if reranked else ()),
        "section > ... > entry",
    )
    chunks = f", {clf.last_chunks} chunks" if clf.last_chunks > 1 else ""
    table = Table(*columns, title=f"IPC {clf.version} {clf.lang}{chunks}")
    for i, m in enumerate(matches, 1):
        judge = (f"{m.judge:.2f}",) if reranked else ()
        table.add_row(str(i), m.pretty, f"{m.score:.3f}", f"{m.similarity:.3f}", *judge, m.text)
    console.print(table)


@app.command()
def show(
    symbol: str = typer.Argument(..., help="e.g. 'A01B 1/02' or A01B0001020000"),
    version: str = typer.Option("latest"),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
):
    """Print the symbol path and the full text an entry was embedded with."""
    from .classifier import IpcClassifier
    from .scheme import normalize_symbol

    clf = IpcClassifier(version, lang=lang, model=model)
    node = clf.index.node(normalize_symbol(symbol))
    console.print(f"[bold]{format_symbol(node.symbol)}[/] ({node.level}, depth {node.depth})")
    console.print(" | ".join(format_symbol(s) for s in clf.index.scheme.path_of(node.symbol)))
    console.print(clf.index.text_of(node.symbol))


@app.command("rpi")
def rpi_cases(
    issue: int = typer.Argument(..., help="RPI issue number, e.g. 2905"),
    out: Path = typer.Option(None, help="JSONL path; default evals/rpi_<issue>.jsonl"),
    require_abstract: bool = typer.Option(False, help="Skip records without INID 57"),
):
    """Download an RPI issue and extract eval cases from dispatches 1.3 and 3.1."""
    from .eval import cases_from_rpi, fetch_rpi, parse_rpi, save_cases

    records = parse_rpi(fetch_rpi(issue))
    cases = cases_from_rpi(records, require_abstract=require_abstract)
    target = out or Path("evals") / f"rpi_{issue}.jsonl"
    save_cases(cases, target)
    with_abs = sum(1 for c in cases if c.abstract)
    console.print(
        f"{len(records)} records, {len(cases)} cases ({with_abs} with abstract) -> {target}"
    )


@app.command("eval")
def eval_cmd(
    cases: Path = typer.Argument(..., help="JSONL produced by `t2ipc rpi`"),
    version: str = typer.Option("latest"),
    level: str = typer.Option("subgroup"),
    top_k: int = typer.Option(10),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    limit: int = typer.Option(None, help="Only the first N cases"),
    show_misses: int = typer.Option(0, help="Print this many misses at the target level"),
    rerank: bool = typer.Option(False, help="Second stage: cross-encoder over the candidates"),
    reranker: str = typer.Option(None, help=f"Reranker spec (default {DEFAULT_RERANKER})"),
):
    """Hit-rate at each hierarchy level against office-assigned symbols."""
    from .classifier import IpcClassifier
    from .eval import evaluate, load_cases

    items = load_cases(cases)[:limit]
    clf = IpcClassifier(version, lang=lang, model=model)
    with console.status("evaluating...") as status:
        result = evaluate(
            items,
            lambda t: clf.classify(t, level=level, top_k=top_k, rerank=reranker or rerank),
            level=level,
            top_k=top_k,
            progress=lambda i, n: status.update(f"evaluating {i}/{n}"),
        )
    console.print(
        f"[bold]{cases.name}[/] n={result.n} version={clf.version} lang={clf.lang} "
        f"model={clf.index.meta.model}"
        + (f" reranker={clf.reranker(reranker or True).name}" if (rerank or reranker) else "")
    )
    console.print(result.table())
    for case_id, gold, preds in result.misses[:show_misses]:
        got = " ".join(format_symbol(p) for p in preds)
        console.print(f"  miss {case_id}: gold {gold} | got {got}")


@app.command()
def download(
    repo: str = typer.Argument(DEFAULT_HF_REPO, help="Hugging Face model repo made by hf-export"),
    revision: str = typer.Option(None, help="Branch, tag or commit"),
):
    """Fetch a prebuilt index and its scheme table from the Hugging Face Hub."""
    from .index.publish import download_from_hf

    with console.status(f"downloading from huggingface.co/{repo}..."):
        path, cfg = download_from_hf(repo, revision=revision)
    console.print(f"ready: {path}")
    console.print(
        f'try: t2ipc classify "..." --version {cfg["version"]} --lang {cfg["lang"]} '
        f"--model {cfg['model']}"
    )


@app.command("hf-export")
def hf_export(
    out: Path = typer.Argument(..., help="Directory to create (becomes the HF model repo)"),
    version: str = typer.Option("latest"),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    repo_id: str = typer.Option(None, help="Hub repo id written into the model card"),
):
    """Assemble a Hugging Face Inference Endpoints repository (handler.py + index)."""
    from .classifier import resolve_built_version
    from .hf import export_hf_repo

    model = model or default_model()
    version = resolve_built_version(version, lang.upper(), model, home())
    path = export_hf_repo(
        out,
        version=version,
        lang=lang,
        model=model,
        repo_id=repo_id or f"<user>/text2ipc-{lang.lower()}",
    )
    console.print(f"HF repo assembled at {path}")
    console.print(f"publish with: huggingface-cli upload <user>/<repo> {path} . --repo-type model")


@app.command("web-export")
def web_export(
    out: Path = typer.Argument(..., help="Directory to create (becomes a static HF Space)"),
    version: str = typer.Option("latest", help="Resolved per language among built indexes"),
    lang: list[str] = typer.Option(["PT"], help="Scheme language; repeat for several"),
    model: str = typer.Option(
        None, help=f"Embedder the indexes were built with (default {WEB_DEFAULT_MODEL})"
    ),
    web_model: str = typer.Option(None, help="transformers.js model id (default: Xenova twin)"),
    web_dtype: str = typer.Option("q8", help="ONNX weights the browser loads: q8, fp16, fp32"),
    repo_id: str = typer.Option(None, help="Space id written into the README"),
    examples: Path = typer.Option(
        None, help="JSONL of eval cases; their title + abstract become the example texts"
    ),
    web_reranker: str = typer.Option(
        WEB_DEFAULT_RERANKER,
        help="transformers.js cross-encoder for the Rerank option; '' for none",
    ),
):
    """Assemble a static Hugging Face Space that classifies in the browser (ADR 0007)."""
    from .web import export_web_demo

    path = export_web_demo(
        out,
        model=model or WEB_DEFAULT_MODEL,
        langs=[lg.upper() for lg in lang],
        version=version,
        repo_id=repo_id or "<user>/text2ipc",
        web_model=web_model,
        web_dtype=web_dtype,
        examples=examples,
        web_reranker=web_reranker or None,
    )
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    console.print(f"static Space assembled at {path} ({total / 1e6:.0f} MB)")
    console.print(
        f"publish with: scripts/publish_space.sh "
        f"(or hf upload <user>/<space> {path} . --repo-type space)"
    )


@app.command("migrate-csv")
def migrate_csv(
    csv_path: Path = typer.Argument(..., help="Legacy ipc_<version>_<lang>_<model>.csv index"),
    delete: bool = typer.Option(False, help="Delete the CSV and its meta.json afterwards"),
):
    """Convert a legacy CSV index into scheme table + parquet index, without embedding."""
    from .index.migrate import migrate_csv_index

    scheme_target, index_target, report = migrate_csv_index(csv_path)
    console.print(report)
    console.print(f"wrote {scheme_target.name} and {index_target.name}")
    if delete:
        csv_path.unlink()
        csv_path.with_suffix(".meta.json").unlink(missing_ok=True)
        console.print("deleted the CSV")


if __name__ == "__main__":
    app()
