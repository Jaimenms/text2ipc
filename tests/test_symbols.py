import pytest

from text2ipc.scheme import format_symbol, level_of_symbol, normalize_symbol


@pytest.mark.parametrize(
    "canonical,pretty",
    [
        ("A01B0001020000", "A01B 1/02"),
        ("A01B0001000000", "A01B 1/00"),
        ("H04L0067100000", "H04L 67/10"),
        ("G06F0009451000", "G06F 9/451"),
        ("A61K0031474500", "A61K 31/4745"),
        ("A01B", "A01B"),
    ],
)
def test_format_and_normalize_roundtrip(canonical, pretty):
    assert format_symbol(canonical) == pretty
    assert normalize_symbol(pretty) == canonical
    assert normalize_symbol(canonical) == canonical


def test_normalize_accepts_compact_form():
    assert normalize_symbol("a01b1/02") == "A01B0001020000"


def test_levels():
    assert level_of_symbol("A") == "section"
    assert level_of_symbol("A01") == "class"
    assert level_of_symbol("A01B") == "subclass"
    assert level_of_symbol("A01B0001000000") == "group"
    assert level_of_symbol("A01B0001020000") == "subgroup"
    with pytest.raises(ValueError):
        normalize_symbol("not a symbol")
