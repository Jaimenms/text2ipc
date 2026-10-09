import json
import sys

from text2ipc.hf import export_hf_repo


def test_export_and_handler_roundtrip(mini_home, tmp_path, monkeypatch):
    repo = export_hf_repo(
        tmp_path / "repo", version="20260101", lang="EN", model="hash:64", root=mini_home
    )

    assert (repo / "handler.py").exists()
    assert (repo / "text2ipc" / "classifier.py").exists()
    assert (repo / "index" / "ipc_20260101_en_hash-64.parquet").exists()
    assert (repo / "scheme" / "ipc_20260101_en.parquet").exists()
    cfg = json.loads((repo / "text2ipc.json").read_text())
    assert cfg == {"version": "20260101", "lang": "EN", "model": "hash:64"}
    card = (repo / "README.md").read_text()
    assert "pipeline_tag: text-classification" in card
    assert "| `rerank` |" in card and "`judge`" in card
    assert 'lang="EN"' in card and "hand hoe with two blades" in card  # examples per language
    assert "base_model:\n  - hash:64\n  - BAAI/bge-reranker-v2-m3\n" in card.replace(
        "base_model:\n  - 64\n", "base_model:\n  - hash:64\n"
    )

    monkeypatch.syspath_prepend(str(repo))
    sys.modules.pop("handler", None)
    import handler

    h = handler.EndpointHandler(str(repo))
    out = h({"inputs": "hand tools spades shovels with teeth", "parameters": {"top_k": 3}})
    assert out[0]["symbol"] == "A01B 1/04" and out[0]["canonical"] == "A01B0001040000"
    assert set(out[0]) >= {"symbol", "level", "score", "similarity", "title", "path"}
    batch = h({"inputs": ["harrows", "network protocols"], "parameters": {"level": "group"}})
    assert [b[0]["symbol"] for b in batch] == ["A01B 19/00", "H04L 67/00"]
    assert h({"inputs": "harrows", "parameters": {"bogus": 1, "top_k": 1}})
