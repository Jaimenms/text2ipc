from text2ipc.scheme import SchemeTable, apply_titles


def test_parse_skips_non_classifiable_and_keeps_order(mini_nodes):
    symbols = [n.symbol for n in mini_nodes]
    assert symbols[:3] == ["A", "A01", "A01B"]
    assert "A01B0001040000" in symbols
    by = {n.symbol: n for n in mini_nodes}
    assert by["A01B0001000000"].level == "group"
    assert by["A01B0001040000"].level == "subgroup"
    assert by["A01B0001040000"].depth == 2
    assert by["A01B0001040000"].parent == "A01B0001020000"
    pos = {s: i for i, s in enumerate(symbols)}
    assert all(pos[n.parent] < pos[n.symbol] for n in mini_nodes if n.parent)


def test_titles_drop_cross_references_and_join_parts(mini_nodes):
    by = {n.symbol: n for n in mini_nodes}
    assert by["A01B"].title == "SOIL WORKING IN AGRICULTURE"
    assert by["A01B0001020000"].title == "Spades; Shovels"
    assert by["A01B0001000000"].title == "Hand tools"


def test_guidance_heading_covers_its_range_only(mini_nodes):
    by = {n.symbol: n for n in mini_nodes}
    assert by["A01B0003000000"].heading == "Ploughs"
    assert by["A01B0019000000"].heading == ""
    assert by["A01B0001000000"].heading == ""


def test_scheme_table_paths_and_text(mini_scheme):
    assert mini_scheme.path_of("A01B0001040000") == (
        "A",
        "A01",
        "A01B",
        "A01B0001000000",
        "A01B0001020000",
        "A01B0001040000",
    )
    assert mini_scheme.text_of("A01B0001040000") == (
        "HUMAN NECESSITIES > AGRICULTURE; FORESTRY > SOIL WORKING IN AGRICULTURE > "
        "Hand tools > Spades; Shovels > with teeth"
    )
    assert "SOIL WORKING IN AGRICULTURE > Ploughs > Ploughs with fixed" in mini_scheme.text_of(
        "A01B0003000000"
    )
    assert mini_scheme.text_of("A") == "HUMAN NECESSITIES"


def test_scheme_table_parquet_roundtrip(mini_scheme, tmp_path):
    path = tmp_path / "ipc_20260101_en.parquet"
    mini_scheme.write(path)
    loaded = SchemeTable.read(path)
    assert loaded.nodes == mini_scheme.nodes
    assert loaded.paths == mini_scheme.paths
    assert loaded.texts() == mini_scheme.texts()


def test_overlay_translates_titles_and_paths(mini_nodes):
    translated = apply_titles(
        mini_nodes, {"A01B0001000000": "Ferramentas manuais", "A01B 1/04": "com dentes"}
    )
    table = SchemeTable.from_nodes(translated)
    assert table.node("A01B0001000000").title == "Ferramentas manuais"
    assert table.text_of("A01B0001040000").endswith(
        "Ferramentas manuais > Spades; Shovels > com dentes"
    )
    assert [n.symbol for n in translated] == [n.symbol for n in mini_nodes]
