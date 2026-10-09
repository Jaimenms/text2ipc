"""The static browser demo: export layout, int8 vectors, and parity between the
JavaScript scorer and the Python one (run under Node when it is installed)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from text2ipc.search import Beam, SearchParams, search
from text2ipc.web import dequantize_int8, export_web_demo, quantize_int8

NODE = shutil.which("node")
PARITY = Path(__file__).with_name("js_parity.mjs")


def test_int8_roundtrip_keeps_cosine(mini_index):
    q, scales = quantize_int8(mini_index.vectors)
    back = dequantize_int8(q, scales)
    cos = np.sum(back * mini_index.vectors, axis=1) / np.linalg.norm(back, axis=1)
    assert q.dtype == np.int8 and scales.shape == (len(mini_index),)
    assert cos.min() > 0.999


def test_export_layout(mini_home, tmp_path):
    out = export_web_demo(
        tmp_path / "space", model="hash:64", langs=["EN"], root=mini_home, web_model="test/model"
    )
    for name in ("index.html", "app.js", "scorer.js", "README.md", "manifest.json"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["web_model"] == "test/model" and manifest["dim"] == 64
    assert manifest["query_prefix"] == ""
    (entry,) = manifest["indexes"]
    assert entry["lang"] == "EN" and entry["version"] == "20260101"
    scheme = json.loads((out / entry["scheme"]).read_text())
    rows = entry["rows"]
    assert all(len(scheme[c]) == rows for c in scheme)
    assert scheme["parent"][0] == -1 and all(p < i for i, p in enumerate(scheme["parent"]))
    assert (out / entry["vectors"]).stat().st_size == rows * 4 + rows * 64
    assert "sdk: static" in (out / "README.md").read_text()


def test_export_writes_examples(mini_home, tmp_path):
    cases = tmp_path / "ex.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "rpi1:X",
                "text": "a",
                "ipc": ["A01B0001040000"],
                "title": "PÁ COM DENTES",
                "abstract": "Uma pá com dentes.",
                "source": "RPI 1",
            }
        )
        + "\n"
    )
    out = export_web_demo(
        tmp_path / "space",
        model="hash:64",
        langs=["EN"],
        root=mini_home,
        web_model="test/model",
        examples=cases,
    )
    (ex,) = json.loads((out / "examples.json").read_text())
    assert ex["title"] == "PÁ COM DENTES" and ex["ipc"] == ["A01B0001040000"]
    cases.write_text(json.dumps({"id": "rpi1:Y", "text": "b", "ipc": ["A"], "title": "T"}) + "\n")
    with pytest.raises(ValueError, match="title and an abstract"):
        export_web_demo(
            tmp_path / "space2",
            model="hash:64",
            langs=["EN"],
            root=mini_home,
            web_model="test/model",
            examples=cases,
        )


def test_export_rejects_unknown_browser_model(mini_home, tmp_path):
    with pytest.raises(ValueError, match="browser model"):
        export_web_demo(tmp_path / "space", model="hash:64", langs=["EN"], root=mini_home)


QUERIES = [
    "hand tools spades shovels with teeth",
    "protocols network nodes",
    "harrows",
    "harrows non-rotating tools",
    "soil working agriculture",
    "hoes hand cultivators with two or more blades",
    "network protocols digital information",
]
PARAMS = [
    SearchParams(),
    SearchParams(top_k=3),
    SearchParams(level="section"),
    SearchParams(level="class"),
    SearchParams(level="subclass"),
    SearchParams(level="group"),
    SearchParams(level="group", gap=0.0),
    SearchParams(level="group", beam=Beam(section=1, class_=1, subclass=1, group=1)),
    SearchParams(level="auto"),
    SearchParams(level="auto", auto_margin=0.0),
    SearchParams(dedupe_branches=False),
]


def _js_params(p: SearchParams) -> dict:
    return {
        "level": p.level,
        "topK": p.top_k,
        "gap": p.gap,
        "autoMargin": p.auto_margin,
        "autoRoots": p.auto_roots,
        "dedupeBranches": p.dedupe_branches,
        "weights": asdict(p.weights),
        "beam": {
            "section": p.beam.section,
            "class": p.beam.class_,
            "subclass": p.beam.subclass,
            "group": p.beam.group,
        },
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_scorer_matches_python(mini_home, mini_index, embedder, tmp_path):
    out = export_web_demo(
        tmp_path / "space",
        model="hash:64",
        langs=["EN"],
        root=mini_home,
        web_model="test/model",
        encoding="float32",
    )
    cases, expected = [], []
    for text in QUERIES:
        vec = embedder.embed_query(text)
        for p in PARAMS:
            cases.append({"lang": "EN", "query": vec.tolist(), "params": _js_params(p)})
            expected.append(search(mini_index, vec, p))
    stack = embedder.embed_queries(QUERIES[:3])  # a chunked long text: max over chunks
    for p in (SearchParams(), SearchParams(level="group"), SearchParams(level="auto")):
        cases.append({"lang": "EN", "query": stack.tolist(), "params": _js_params(p)})
        expected.append(search(mini_index, stack, p))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    run = subprocess.run(
        [NODE, str(PARITY), str(out), str(cases_path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    assert len(got) == len(expected)
    for case, py, js in zip(cases, expected, got, strict=True):
        label = f"{case['params']}"
        assert [m.symbol for m in py] == [m["symbol"] for m in js], label
        for a, b in zip(py, js, strict=True):
            assert (a.pretty, a.level, a.depth, a.title, a.text) == (
                b["pretty"],
                b["level"],
                b["depth"],
                b["title"],
                b["text"],
            ), label
            for field in ("score", "similarity", "path_support", "subtree_support"):
                assert abs(getattr(a, field) - b[field]) < 1e-5, (label, field)


SPLIT_TEXTS = [
    "Pá com dentes. Enxada manual com duas lâminas! Grade de discos; grade de dentes.\n\n"
    "Segundo parágrafo: ferramentas de mão. Fim.",
    " ".join(f"palavra{i}" for i in range(23)),
    "Uma frase só.",
    "A. B. C. D. E. F. G.",
]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_split_matches_python(mini_home, tmp_path):
    from text2ipc.chunking import split_text

    out = export_web_demo(
        tmp_path / "space", model="hash:64", langs=["EN"], root=mini_home, web_model="t/m"
    )
    count = lambda t: len(t.split())  # noqa: E731
    cases, expected = [], []
    for text in SPLIT_TEXTS:
        for max_tokens, overlap in ((6, 1), (10, 0), (4, 2)):
            cases.append({"split": {"text": text, "max_tokens": max_tokens, "overlap": overlap}})
            expected.append(split_text(text, max_tokens, count, overlap=overlap))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    run = subprocess.run(
        [NODE, str(PARITY), str(out), str(cases_path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == expected


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_fusion_matches_python(mini_home, tmp_path):
    from text2ipc.rerank import fuse

    out = export_web_demo(
        tmp_path / "space", model="hash:64", langs=["EN"], root=mini_home, web_model="t/m"
    )
    scores = [0.9, 0.85, 0.8, 0.5]
    logits = [-3.0, 0.0, 2.5, 20.0]
    cases = [
        {"fuse": {"scores": scores, "logits": logits, "how": h}}
        for h in ("blend", "judge", "product")
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    run = subprocess.run(
        [NODE, str(PARITY), str(out), str(cases_path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    for case, got in zip(cases, json.loads(run.stdout), strict=True):
        expected = fuse(scores, logits, case["fuse"]["how"])
        assert all(abs(a - b) < 1e-9 for a, b in zip(expected, got, strict=True))
