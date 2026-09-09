from pathlib import Path

from text2ipc.eval import cases_from_rpi, evaluate, load_cases, parse_rpi, save_cases
from text2ipc.eval.harness import truncate
from text2ipc.search import SearchParams, search

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_rpi_keeps_published_dispatches_only():
    records = parse_rpi(FIXTURES / "mini_rpi.xml")
    assert [r["dispatch"] for r in records] == ["3.1", "1.3"]
    assert records[0]["title"] == "PÁ COM DENTES PARA JARDINAGEM"
    assert records[0]["abstract"] == "Uma pá manual com dentes."
    assert records[0]["ipc"] == ["A01B 1/04", "A01B 3/00"]


def test_cases_from_rpi_and_roundtrip(tmp_path):
    cases = cases_from_rpi(parse_rpi(FIXTURES / "mini_rpi.xml"))
    assert len(cases) == 2
    assert cases[0].text == "Uma pá manual com dentes."
    assert cases[0].ipc == ("A01B0001040000", "A01B0003000000")
    assert cases[1].text == "PROTOCOLO DE REDE DISTRIBUÍDO" and "title-only" in cases[1].tags
    save_cases(cases, tmp_path / "c.jsonl")
    assert load_cases(tmp_path / "c.jsonl") == cases
    assert len(cases_from_rpi(parse_rpi(FIXTURES / "mini_rpi.xml"), require_abstract=True)) == 1


def test_truncate():
    assert truncate("A01B0001040000", "section") == "A"
    assert truncate("A01B0001040000", "class") == "A01"
    assert truncate("A01B0001040000", "subclass") == "A01B"
    assert truncate("A01B0001040000", "group") == "A01B0001000000"
    assert truncate("A01B0001040000", "subgroup") == "A01B0001040000"


def test_evaluate_hit_rates(mini_index, embedder):
    cases = cases_from_rpi(parse_rpi(FIXTURES / "mini_rpi.xml"))
    # English queries against the English mini scheme so the hash embedder can match.
    cases = [
        cases[0].__class__(**{**cases[0].__dict__, "text": "hand tools spades shovels with teeth"}),
        cases[1].__class__(
            **{**cases[1].__dict__, "text": "network protocols application distributed nodes"}
        ),
    ]
    result = evaluate(
        cases,
        lambda t: search(mini_index, embedder.embed_query(t), SearchParams(top_k=5)),
        top_k=5,
    )
    assert result.n == 2
    assert result.rate("subclass", 1) == 1.0
    assert result.rate("subgroup", 1) == 1.0
    assert result.mrr("group") == 1.0
    assert "hit@1" in result.table()


def test_parse_rpi_text_format():
    records = parse_rpi(FIXTURES / "mini_rpi.txt")
    assert [r["dispatch"] for r in records] == ["1.3", "3.1"]
    first = records[0]
    assert first["issue"] == "2100"
    assert first["number"] == "PI 0309892-3 A2"
    assert first["ipc"] == ["C07D 239/36", "A61K 31/505"]
    # the abstract drops the repeated title and joins the continuation line
    assert first["abstract"] == "Compostos úteis para tratar doenças proliferativas são divulgados."
    assert records[1]["abstract"] == "" and records[1]["ipc"] == ["A01B 1/04"]
    cases = cases_from_rpi(records, require_abstract=True)
    assert len(cases) == 1 and cases[0].ipc == ("C07D0239360000", "A61K0031505000")


def test_load_many_stratified_sample(tmp_path):
    from text2ipc.eval import EvalCase, load_many, save_cases

    cases = [
        EvalCase(id=f"a{i}", text="x", ipc=("A01B0001000000",), abstract="x") for i in range(30)
    ] + [EvalCase(id=f"t{i}", text="y", ipc=("A01B0001000000",)) for i in range(5)]
    save_cases(cases[:20], tmp_path / "rpi_1.jsonl")
    save_cases(cases[20:], tmp_path / "rpi_2.jsonl")
    picked = load_many(str(tmp_path / "rpi_*.jsonl"), sample=20, seed=1)
    assert len(picked) == 20
    assert sum(1 for c in picked if c.abstract) == 15  # 5 title-only is all there is
    assert load_many(str(tmp_path / "rpi_*.jsonl")) == cases
    assert load_many(str(tmp_path / "rpi_*.jsonl"), sample=4, seed=2) != load_many(
        str(tmp_path / "rpi_*.jsonl"), sample=4, seed=3
    )
