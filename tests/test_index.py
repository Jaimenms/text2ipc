import numpy as np
import pytest

from text2ipc.embeddings.hashing import HashEmbedder
from text2ipc.index import (
    IpcIndex,
    build_index,
    find_previous_index,
    index_path,
    scheme_table_path,
)
from text2ipc.index.store import vector_columns
from text2ipc.scheme import SchemeTable, apply_titles
from text2ipc.scheme.model import IpcNode


def test_parquet_roundtrip(mini_index, mini_home):
    path = index_path("20260101", "EN", "hash:64", mini_home)
    loaded = IpcIndex.read(path, scheme_table_path("20260101", "EN", mini_home))
    assert loaded.symbols == mini_index.symbols
    assert loaded.texts == mini_index.texts
    assert loaded.hashes() == mini_index.hashes()
    np.testing.assert_allclose(loaded.vectors, mini_index.vectors, atol=1e-6)
    assert loaded.meta.model == "hash:64" and loaded.meta.version == "20260101"
    assert IpcIndex.read_meta(path).dim == 64
    assert vector_columns(64)[:2] == ["e000", "e001"] and vector_columns(768)[-1] == "e767"


def test_index_refuses_mismatched_scheme(mini_index, mini_home, mini_nodes):
    other = SchemeTable.from_nodes(mini_nodes[:-1])
    with pytest.raises(ValueError):
        IpcIndex.read(index_path("20260101", "EN", "hash:64", mini_home), other)


def test_build_reuses_unchanged_vectors(mini_nodes, embedder):
    v1, r1 = build_index(
        SchemeTable.from_nodes(mini_nodes), embedder, version="20260101", lang="EN"
    )
    assert r1.previous is None and r1.computed == len(mini_nodes)

    changed = apply_titles(mini_nodes, {"A01B0001020000": "Spades; Shovels; Scoops"})
    removed = changed[-1]
    changed = [n for n in changed if n is not removed]
    changed.append(
        IpcNode(
            symbol="A01B0019100000",
            level="subgroup",
            depth=1,
            parent="A01B0019000000",
            title="with spring tines",
        )
    )
    v2, r2 = build_index(
        SchemeTable.from_nodes(changed), embedder, version="20270101", lang="EN", previous=v1
    )
    assert r2.previous == "20260101"
    assert set(r2.changed) == {"A01B0001020000", "A01B0001040000"}
    assert r2.added == ["A01B0019100000"]
    assert r2.removed == [removed.symbol]
    assert r2.reused == len(changed) - 3 and r2.computed == 3
    for sym in ("A01B", "A01B0003000000"):
        np.testing.assert_array_equal(v2.vectors[v2.position[sym]], v1.vectors[v1.position[sym]])


def test_build_reuses_from_disk(mini_home, mini_nodes, embedder):
    prev = (
        index_path("20260101", "EN", "hash:64", mini_home),
        scheme_table_path("20260101", "EN", mini_home),
    )
    v2, r2 = build_index(
        SchemeTable.from_nodes(mini_nodes), embedder, version="20270101", lang="EN", previous=prev
    )
    assert r2.reused == len(mini_nodes) and r2.computed == 0


def test_build_refuses_other_model(mini_index, mini_scheme):
    with pytest.raises(ValueError):
        build_index(
            mini_scheme, HashEmbedder(32), version="20270101", lang="EN", previous=mini_index
        )


def test_previous_index_discovery(mini_index, tmp_path):
    mini_index.write(index_path("20250101", "EN", "hash:64", tmp_path))
    mini_index.write(index_path("20260101", "EN", "hash:64", tmp_path))
    mini_index.write(index_path("20260101", "FR", "hash:64", tmp_path))
    prev = find_previous_index("20270101", "EN", "hash:64", tmp_path)
    assert prev is not None and prev.name == "ipc_20260101_en_hash-64.parquet"
    assert find_previous_index("20250101", "EN", "hash:64", tmp_path) is None
    assert find_previous_index("20270101", "EN", "st:other", tmp_path) is None
