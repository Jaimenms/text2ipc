from text2ipc.scheme.inpi import clean_title

GROUP = (
    '<div>Implementos manuais<span> (aparadores de bordaduras para gramados <a rel="symbol" '
    'href="#">A01G 3/06</a>)</span> <a rel="versions" data-content="1<br>2006.01" href="#">'
    "<strong>[2006.01]</strong></a></div>"
)
SUBGROUP = (
    '<div class="lvl  dot-1 last_child"><div>•</div></div><div class="lvl  dot_last no_child">'
    '<div>•</div></div><div class="txt lvl2">com dentes <a rel="versions" data-content="1<br>'
    '2006.01" href="#"><strong>[2006.01]</strong></a></div>'
)
SUBCLASS = (
    "<div>PLANTIO; SEMEADURA; FERTILIZAÇÃO<span> (peças, detalhes ou acessórios "
    '<a rel="symbol" href="#">A01B 51/00</a>-<a rel="symbol" href="#">A01B 75/00</a>)</span> </div>'
)


def test_clean_title_strips_refs_versions_and_markup():
    assert clean_title(GROUP, "m") == "Implementos manuais"
    assert clean_title(SUBGROUP, "2") == "com dentes"
    assert clean_title(SUBCLASS, "u") == "PLANTIO; SEMEADURA; FERTILIZAÇÃO"
    assert clean_title("<div>SEÇÃO A — NECESSIDADES HUMANAS</div>", "s") == "NECESSIDADES HUMANAS"
    assert clean_title("<div>Arados </div>", "g") == "Arados"
    bare = '<div class="txt lvl9">o dado de corrente <strong>[2016.01]</strong></div>'
    assert clean_title(bare, "9") == "o dado de corrente"
