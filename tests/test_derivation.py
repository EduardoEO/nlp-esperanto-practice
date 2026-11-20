"""Pruebas unitarias para la descomposición derivativa de afijos y el modo ``root``."""
import pytest

from esperanto_nlp import analyze, load
from esperanto_nlp._derivation import decompose


@pytest.mark.parametrize("stem, expected", [
    ("lernej", ((), "lern", ("ej",))),
    ("malbonig", (("mal",), "bon", ("ig",))),
    ("fizikist", ((), "fizik", ("ist",))),
    ("instruistin", ((), "instru", ("ist", "in"))),
    ("sanigej", ((), "san", ("ig", "ej"))),          # sufijos encadenados
    ("belec", ((), "bel", ("ec",))),
    ("kapitalism", ((), "kapital", ("ism",))),
])
def test_decompose_known_derivations(stem, expected):
    """Verifica el aislamiento correcto de prefijos y sufijos simples o encadenados."""
    assert decompose(stem) == expected


@pytest.mark.parametrize("stem", [
    "molekul", "planet", "oksid", "vitamin", "sistem", "polar", "nuklear",   # vocabulario científico
    "ekzist", "resist", "koleg",                                              # excepciones (NO_SPLIT)
    "malari",                                                                 # mal- que no es prefijo
    "reg", "lig", "plej",                                                     # raíces demasiado cortas
])
def test_decompose_leaves_these_untouched(stem):
    """Verifica que no se fragmenten raíces científicas, excepciones ``NO_SPLIT`` ni raíces cortas."""
    assert decompose(stem) == ((), stem, ())


def test_analyze_exposes_base_and_affixes():
    """Verifica que ``analyze`` conserve el lema intacto mientras expone ``base`` y ``affixes``."""
    a = analyze("malbonigoj")
    assert (a.root, a.base, a.affixes) == ("malbonig", "bon", ("mal-", "-ig"))
    assert a.lemma == "malbonigo"          # el lema NO pierde los afijos


def test_closed_class_and_unknown_have_no_affixes():
    """Verifica que las palabras cerradas y los nombres propios no sufran descomposición."""
    assert analyze("kaj").affixes == () and analyze("kaj").base == "kaj"
    assert analyze("Zamenhof").affixes == ()


def test_pipeline_fills_root_and_affixes():
    """Verifica que el pipeline completo rellene ``root_``, ``affixes`` y ``to_dicts()``."""
    doc = load()("La lernejoj kaj la fizikistoj.")
    by_text = {t.text: t for t in doc}
    assert by_text["lernejoj"].root_ == "lern" and by_text["lernejoj"].affixes == ("-ej",)
    assert by_text["fizikistoj"].root_ == "fizik"
    assert by_text["kaj"].root_ == "kaj"
    assert "root" in doc.to_dicts()[0]


def test_sklearn_root_mode_merges_word_family():
    """Verifica que ``feature='root'`` agrupe familias derivadas (``fizikisto``/``fiziko``)."""
    pytest.importorskip("sklearn")
    wrapper = pytest.importorskip("esperanto_nlp.sklearn_wrapper")
    EsperantoPreprocessor = wrapper.EsperantoPreprocessor
    out = EsperantoPreprocessor(feature="root", drop_stop=True).fit_transform(
        ["La fizikisto laboras.", "Fiziko estas scienco.", "Fizikaj leĝoj."])
    # fizikisto / fiziko / fizika -> comparten raíz "fizik" (fiziko y fizika ya la comparten por lema/stem)
    assert all("fizik" in o.split() for o in out)
