"""Pruebas unitarias para el analizador morfológico y el léxico cerrado."""
import pytest

from esperanto_nlp import analyze
from esperanto_nlp.lexicon import DEFAULT_STOPWORDS, LEXICON, is_compound_numeral


@pytest.mark.parametrize("word, pos, lemma", [
    # terminaciones básicas
    ("hundo", "NOUN", "hundo"), ("hundojn", "NOUN", "hundo"), ("hundoj", "NOUN", "hundo"),
    ("granda", "ADJ", "granda"), ("grandajn", "ADJ", "granda"),
    ("rapide", "ADV", "rapide"), ("hejmen", "ADV", "hejme"),
    ("kuri", "VERB", "kuri"), ("kuras", "VERB", "kuri"), ("kuris", "VERB", "kuri"),
    ("kuros", "VERB", "kuri"), ("kurus", "VERB", "kuri"), ("kuru", "VERB", "kuri"),
    # TRAMPAS: palabras cerradas que parecen otra cosa por su terminación
    ("kio", "PRON", "kio"), ("tion", "PRON", "tio"), ("kiujn", "PRON", "kiu"),
    ("kian", "DET", "kia"), ("tie", "ADV", "tie"), ("kiam", "ADV", "kiam"),
    ("tre", "ADV", "tre"), ("ĉe", "ADP", "ĉe"), ("de", "ADP", "de"), ("ĉu", "PART", "ĉu"),
    ("ankaŭ", "ADV", "ankaŭ"), ("mi", "PRON", "mi"), ("min", "PRON", "mi"),
    ("mia", "DET", "mia"), ("miajn", "DET", "mia"), ("la", "DET", "la"),
    ("kaj", "CCONJ", "kaj"), ("plus", "ADP", "plus"),
    # esti -> AUX
    ("estas", "AUX", "esti"), ("estis", "AUX", "esti"), ("esti", "AUX", "esti"),
    # participios
    ("kuranta", "ADJ", "kuri"), ("kurinta", "ADJ", "kuri"), ("kuronta", "ADJ", "kuri"),
    ("skribata", "ADJ", "skribi"), ("loĝantoj", "NOUN", "loĝi"), ("prezidanto", "NOUN", "prezidi"),
    ("rapidante", "ADV", "rapidi"),
    # falsos participios que NO deben serlo
    ("kato", "NOUN", "kato"), ("dato", "NOUN", "dato"), ("ŝtato", "NOUN", "ŝtato"),
    ("elefanto", "NOUN", "elefanto"), ("kandidato", "NOUN", "kandidato"),
    # números
    ("dudek", "NUM", "dudek"), ("tricent", "NUM", "tricent"), ("unu", "NUM", "unu"),
    ("dua", "ADJ", "dua"), ("dudeka", "ADJ", "dudeka"), ("1950", "NUM", "1950"),
    # x-system y mayúsculas al inicio
    ("cxu", "PART", "ĉu"), ("gxi", "PRON", "ĝi"),
])
def test_analysis(word, pos, lemma):
    """Verifica la asignación de POS y lema en terminaciones regulares, léxico, participios y números."""
    a = analyze(word, sentence_start=True)
    assert (a.pos, a.lemma) == (pos, lemma), a


def test_features():
    """Verifica la extracción de rasgos morfológicos (número, caso, tiempo, modo y voz)."""
    assert dict(analyze("hundojn").feats) == {"Number": "Plur", "Case": "Acc"}
    assert dict(analyze("kuris").feats) == {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"}
    assert dict(analyze("kurus").feats)["Mood"] == "Cnd"
    assert dict(analyze("kuri").feats) == {"VerbForm": "Inf"}
    f = dict(analyze("skribata").feats)
    assert (f["Voice"], f["Tense"], f["VerbForm"]) == ("Pass", "Pres", "Part")
    assert dict(analyze("kurinta").feats)["Voice"] == "Act"


def test_roots():
    """Verifica que ``root`` elimine las terminaciones flexivas de sustantivos y verbos."""
    assert analyze("hundojn").root == "hund"
    assert analyze("kuris").root == "kur"


def test_proper_nouns():
    """Verifica la detección de nombres propios, siglas, extranjerismos y titulares en mayúsculas."""
    assert analyze("Parizo").pos == "PROPN"                        # mayúscula a mitad de frase
    assert analyze("Parizon").lemma == "Parizo"
    assert analyze("Hundo", sentence_start=True).pos == "NOUN"     # a inicio de frase NO
    assert analyze("Zamenhof", sentence_start=True).pos == "PROPN" # desconocida + mayúscula
    assert analyze("ONU").pos == "PROPN" and analyze("UEFA").pos == "PROPN"
    assert analyze("Washington").pos == "PROPN"                    # letras no esperantistas
    assert analyze("pizzy").pos == "X"
    assert analyze("FUTBALO").pos == "NOUN"                        # titular en mayúsculas


def test_sentence_start_matters_for_capitals():
    """Verifica que una palabra regular con mayúscula sea PROPN solo fuera de inicio de frase."""
    assert analyze("Kato", sentence_start=False).pos == "PROPN"
    assert analyze("Kato", sentence_start=True).pos == "NOUN"


def test_punct_and_empty():
    """Verifica el comportamiento de ``analyze`` ante cadenas vacías y signos de puntuación."""
    assert analyze("").pos == "X"
    assert analyze("...").pos == "PUNCT"


def test_correlatives_are_generated():
    """Verifica la generación sistemática de la tabla de correlativos en ``LEXICON``."""
    corr = [f for f, e in LEXICON.items() if e.pos in {"PRON", "DET", "ADV"} and len(f) >= 2
            and f.startswith(("ki", "ti", "ĉi", "neni"))]
    assert {"kiu", "kiuj", "kiujn", "kiel", "tiam", "ĉiam", "neniam", "nenies", "ĉiom"} <= set(corr)


def test_compound_numerals():
    """Verifica el reconocimiento por programación dinámica de numerales compuestos."""
    assert is_compound_numeral("dudek") and is_compound_numeral("kvarcent")
    assert not is_compound_numeral("hundo") and not is_compound_numeral("")


def test_default_stopwords_contents():
    """Verifica que ``DEFAULT_STOPWORDS`` incluya palabras funcionales y excluya contenido léxico."""
    for w in ("la", "kaj", "de", "kiu", "estas", "mi", "ankaŭ", "tre"):
        assert w in DEFAULT_STOPWORDS
    for w in ("hundo", "futbalo", "dudek"):
        assert w not in DEFAULT_STOPWORDS


@pytest.mark.parametrize("word, lemma, stem", [
    ("Parizo", "Parizo", "pariz"),
    ("Parizojn", "Parizo", "pariz"),
    ("Jupitero", "Jupitero", "jupiter"),
    ("Einstein", "Einstein", "einstein"),     # sin -o: la raíz es el propio nombre en minúscula
])
def test_proper_noun_stem_is_lowercase_without_final_o(word, lemma, stem):
    """Verifica que la raíz de un PROPN se normalice en minúscula y sin ``-o`` final."""
    a = analyze(word, sentence_start=False)
    assert a.pos == "PROPN" and a.lemma == lemma and a.root == stem
