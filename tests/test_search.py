from text2ipc.search import Beam, SearchParams, Weights, search


def q(embedder, text):
    return embedder.embed_query(text)


def test_subgroup_query_finds_the_leaf(mini_index, embedder):
    hits = search(mini_index, q(embedder, "hand tools spades shovels with teeth"))
    assert hits[0].symbol == "A01B0001040000"
    assert hits[0].pretty == "A01B 1/04"
    assert all(h.level == "subgroup" for h in hits)


def test_level_restricts_results(mini_index, embedder):
    for level in ("section", "class", "subclass", "group"):
        hits = search(mini_index, q(embedder, "protocols network nodes"), SearchParams(level=level))
        assert hits and all(h.level == level for h in hits)
    groups = search(mini_index, q(embedder, "protocols network nodes"), SearchParams(level="group"))
    assert groups[0].symbol == "H04L0067000000"


def test_beam_prunes_unrelated_sections(mini_index, embedder):
    params = SearchParams(level="group", beam=Beam(section=1, class_=1, subclass=1, group=1))
    hits = search(mini_index, q(embedder, "network protocols digital information"), params)
    assert {h.symbol[0] for h in hits} == {"H"}


def test_scores_are_composed_and_ordered(mini_index, embedder):
    hits = search(
        mini_index, q(embedder, "harrows non-rotating tools"), SearchParams(level="group")
    )
    assert hits[0].symbol == "A01B0019000000"
    assert hits == sorted(hits, key=lambda h: -h.score)
    top = hits[0]
    w = Weights()
    expected = w.own * top.similarity + w.path * top.path_support + w.subtree * top.subtree_support
    assert abs(top.score - expected) < 1e-5
    assert top.subtree_support >= top.similarity


def test_gap_filters_weak_results(mini_index, embedder):
    hits = search(mini_index, q(embedder, "harrows"), SearchParams(level="group", gap=0.0))
    assert len(hits) == 1 and hits[0].symbol == "A01B0019000000"


def test_auto_level_descends_only_while_supported(mini_index, embedder):
    deep = search(mini_index, q(embedder, "spades shovels with teeth"), SearchParams(level="auto"))
    assert deep[0].symbol == "A01B0001040000"
    shallow = search(
        mini_index,
        q(embedder, "soil working agriculture"),
        SearchParams(level="auto", auto_margin=0.0),
    )
    assert shallow[0].symbol == "A01B"


def test_normalize_query_lowercases_shouting_titles():
    from text2ipc.classifier import normalize_query

    assert normalize_query("  PÁ COM   DENTES ") == "pá com dentes"
    assert normalize_query("A hand hoe with DNA marker") == "A hand hoe with DNA marker"


def test_branch_dedupe_keeps_only_distinct_branches(mini_index, embedder):
    # Both hoes subgroups and their parent group share the branch A01B 1/06; only the
    # best-ranked survivor of that branch may stay, so ranked results never contain
    # an ancestor/descendant pair.
    hits = search(mini_index, q(embedder, "hoes hand cultivators with two or more blades"))
    symbols = [h.symbol for h in hits]
    assert "A01B0001100000" in symbols
    paths = {s: set(mini_index.scheme.path_of(s)) for s in symbols}
    for a in symbols:
        for b in symbols:
            if a != b:
                assert a not in paths[b] and b not in paths[a]
    # the rule is off when asked
    raw = search(
        mini_index,
        q(embedder, "hoes hand cultivators with two or more blades"),
        SearchParams(dedupe_branches=False),
    )
    assert len(raw) >= len(hits)


def test_branch_dedupe_across_levels_in_auto_mode(mini_index, embedder):
    hits = search(mini_index, q(embedder, "spades shovels with teeth"), SearchParams(level="auto"))
    symbols = [h.symbol for h in hits]
    assert symbols[0] == "A01B0001040000"
    assert "A01B0001020000" not in symbols and "A01B0001000000" not in symbols
