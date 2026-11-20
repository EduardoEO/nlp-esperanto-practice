"""Ayudas en español: cobertura del glosario y explicaciones de palabras."""
import pytest

import esperanto_nlp
from esperanto_nlp.glosario_es import (GLOSARIO, POS_ES, con_glosa, descomponer, explicar_token, glosa,
                                       glosas_de_parrafo, pos_es)
from esperanto_nlp.lexicon import LEXICON

NLP = esperanto_nlp.load()


def tokens(texto):
    """Auxiliar que devuelve un mapa ``{token.text: Token}`` tras procesar ``texto``."""
    return {t.text: t for t in NLP(texto)}


def test_every_closed_class_lemma_has_a_gloss():
    """Si alguien añade una palabra gramatical a la librería, debe añadir también su significado."""
    sin_glosa = sorted({e.lemma for e in LEXICON.values() if not glosa(e.lemma)})
    assert sin_glosa == []


def test_every_pos_tag_has_a_spanish_name():
    """Verifica que toda etiqueta POS producida tenga traducción en ``POS_ES`` y fallback seguro."""
    etiquetas = {t.pos_ for t in NLP("La fizikistoj mezuris 3 km. Zamenhof legis https://x.org ekz. jes!")}
    assert etiquetas <= set(POS_ES)
    assert pos_es("NOUN") == "sustantivo" and pos_es("ZZZ") == "ZZZ"      # desconocida: se devuelve tal cual


@pytest.mark.parametrize("palabra, esperado", [("kaj", "y"), ("kio", "¿qué? / lo que"), ("neniam", "nunca"),
                                               ("KAJ", "y")])
def test_gloss_examples_and_case_insensitive(palabra, esperado):
    """Verifica la consulta de glosas conocidas sin sensibilidad a mayúsculas."""
    assert glosa(palabra) == esperado


def test_unknown_word_has_no_gloss_and_is_not_invented():
    """Verifica que ``glosa`` devuelva ``None`` ante palabras desconocidas o vacías."""
    assert glosa("zxqwv") is None and glosa("") is None


def test_con_glosa_handles_multiword_terms():
    """Verifica que ``con_glosa`` anote n-gramas palabra a palabra dejando intactas las ajenas."""
    assert con_glosa("specio homo") == "specio (especie) homo (ser humano)"
    assert con_glosa("palabrainventada") == "palabrainventada"


def test_decompose_plural_noun_with_suffix():
    """Verifica el desglose morfológico de un sustantivo plural con sufijo derivativo."""
    piezas = [p for p, _ in descomponer(tokens("fizikistoj")["fizikistoj"])]
    assert piezas == ["fizik", "-ist", "-o", "-j"]


def test_decompose_prefix_suffix_participle():
    """Verifica el desglose completo con prefijo, sufijo derivativo, participio y plural."""
    piezas = [p for p, _ in descomponer(tokens("malbonigitaj")["malbonigitaj"])]
    assert piezas == ["mal-", "bon", "-ig", "-it-", "-a", "-j"]


def test_decompose_verb_and_accusative():
    """Verifica el desglose de terminación verbal y de caso acusativo en sustantivos."""
    assert [p for p, _ in descomponer(tokens("mezuris")["mezuris"])] == ["mezur", "-is"]
    assert [p for p, _ in descomponer(tokens("rapidon")["rapidon"])] == ["rapid", "-o", "-n"]


def test_explain_token_mentions_category_meaning_and_stopword():
    """Verifica que ``explicar_token`` incluya categoría, glosa, stop-word y puntuación."""
    t = tokens("La hundoj kuris kaj.")
    assert "determinante" in explicar_token(t["La"]) and "palabra vacía" in explicar_token(t["La"])
    assert "plural" in explicar_token(t["hundoj"])
    assert explicar_token(t["."]) == "signo de puntuación"


def test_paragraph_hints_list_known_words_without_stopwords():
    """Verifica que ``glosas_de_parrafo`` liste palabras con contenido excluyendo stop-words."""
    pistas = glosas_de_parrafo("La kemia elemento kaj la metalo estas en la perioda tabelo.", nlp=NLP)
    assert "kemia = químico (adj.)" in pistas and "metalo = metal" in pistas
    assert "la = " not in pistas and "kaj = " not in pistas          # palabras vacías excluidas


def test_glossary_is_nonempty_and_values_are_strings():
    """Verifica la integridad mínima de ``GLOSARIO`` (>250 entradas no vacías)."""
    assert len(GLOSARIO) > 250 and all(isinstance(v, str) and v for v in GLOSARIO.values())


def test_con_glosa_never_nests_parentheses():
    """Verifica que ``con_glosa`` aplane paréntesis internos de las glosas en toda la tabla."""
    assert con_glosa("kemia") == "kemia (químico, adj.)"
    assert con_glosa("la") == "la (el / la / los / las, artículo)"
    assert all(con_glosa(w).count("(") == 1 for w in GLOSARIO)
