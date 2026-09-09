import json
from pathlib import Path

from text2ipc.index import publish


def test_download_from_hf_copies_index_and_scheme(mini_home, tmp_path, monkeypatch):
    # Fake Hub: files come from mini_home, laid out like an hf-export repository.
    repo = tmp_path / "repo"
    (repo / "index").mkdir(parents=True)
    (repo / "scheme").mkdir()
    for sub in ("index", "scheme"):
        for f in (mini_home / sub).glob("*.parquet"):
            (repo / sub / f.name).write_bytes(f.read_bytes())
    (repo / "text2ipc.json").write_text(
        json.dumps({"version": "20260101", "lang": "EN", "model": "hash:64"})
    )
    files = ["README.md", "handler.py", "text2ipc.json"] + [
        f"{p.parent.name}/{p.name}" for p in repo.glob("*/*.parquet")
    ]

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            return files

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, revision=None, token=None: str(repo / filename),
    )
    target = tmp_path / "home"
    path, cfg = publish.download_from_hf("someone/text2ipc-en", root=target)
    assert path == target / "index" / "ipc_20260101_en_hash-64.parquet"
    assert (target / "scheme" / "ipc_20260101_en.parquet").exists()
    assert cfg["model"] == "hash:64"
    from text2ipc.classifier import IpcClassifier

    clf = IpcClassifier("latest", lang="EN", model="hash:64", root=Path(target))
    assert clf.classify("harrows", level="group", top_k=1)[0].pretty == "A01B 19/00"
