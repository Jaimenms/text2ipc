from pathlib import Path

import pytest

from text2ipc.embeddings.hashing import HashEmbedder
from text2ipc.index import build_index
from text2ipc.scheme import SchemeTable, parse_scheme

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_nodes():
    return parse_scheme(FIXTURES / "mini_scheme.xml")


@pytest.fixture
def mini_scheme(mini_nodes):
    return SchemeTable.from_nodes(mini_nodes)


@pytest.fixture
def embedder():
    return HashEmbedder(64)


@pytest.fixture
def mini_index(mini_scheme, embedder):
    index, _ = build_index(mini_scheme, embedder, version="20260101", lang="EN")
    return index


@pytest.fixture
def mini_home(mini_index, tmp_path):
    """A text2ipc home with the mini index and scheme table written to disk."""
    from text2ipc.index import index_path, scheme_table_path

    mini_index.scheme.write(scheme_table_path("20260101", "EN", tmp_path))
    mini_index.write(index_path("20260101", "EN", "hash:64", tmp_path))
    return tmp_path
