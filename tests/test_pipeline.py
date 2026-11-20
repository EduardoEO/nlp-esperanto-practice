"""Pruebas de integración para el pipeline ``EsperantoNLP`` y la API de ``Doc``/``Token``."""
import json

import pytest

import esperanto_nlp
from esperanto_nlp import Doc, Token
from esperanto_nlp.stopwords import StopWords


def test_load_and_basic_attributes(nlp):
    """Verifica ``load()``, el orden de componentes y los atributos POS/lema/stop en una frase."""
    doc = nlp("La hundoj kuras rapide.")
    assert isinstance(doc, Doc) and len(doc) == 5
    assert [t.pos_ for t in doc] == ["DET", "NOUN", "VERB", "ADV", "PUNCT"]
    assert [t.lemma_ for t in doc] == ["la", "hundo", "kuri", "rapide", "."]
    assert [t.is_stop for t in doc] == [True, False, False, False, False]
    assert nlp.pipe_names == ["tagger", "lemmatizer", "stopwords"]


def test_doc_api(nlp):
    """Verifica slicing en ``Span``, ``sents``, ``to_list()``, ``to_dicts()`` y ``repr``."""
    doc = nlp("Mi vidis hundon. Ĝi kuris!")
    assert isinstance(doc[0], Token)
    assert doc[0:3].text == "Mi vidis hundon"
    assert [s.text for s in doc.sents] == ["Mi vidis hundon.", "Ĝi kuris!"]
    assert doc.to_list("lemma_", drop_stop=True, drop_punct=True) == ["vidi", "hundo", "kuri"]
    assert doc.to_list("text", keep_pos={"NOUN"}) == ["hundon"]
    assert doc.to_dicts()[2]["lemma"] == "hundo"
    assert "Doc(" in repr(doc)


def test_empty_text(nlp):
    """Verifica que una cadena vacía produzca un ``Doc`` sin tokens ni oraciones."""
    doc = nlp("")
    assert len(doc) == 0 and list(doc.sents) == []


def test_pipe_generator(nlp):
    """Verifica el procesamiento perezoso por lotes mediante el generador ``nlp.pipe()``."""
    docs = list(nlp.pipe(["Mi kuras.", "Vi iras."]))
    assert len(docs) == 2 and docs[1][1].lemma_ == "iri"


def test_x_system_in_pipeline(nlp):
    """Verifica la conversión del sistema-x en ``norm_`` preservando ``text`` original."""
    doc = nlp("Cxu vi parolas Esperanton?")
    assert doc[0].norm_ == "ĉu" and doc[0].text == "Cxu"
    assert doc[3].pos_ == "PROPN" or doc[3].lemma_ in {"Esperanto", "esperanto"}


def test_components_are_independent():
    """Verifica que se pueda instanciar el pipeline con un subconjunto aislado de componentes."""
    only_tagger = esperanto_nlp.load(components=[esperanto_nlp.EsperantoNLP().components[0]])
    doc = only_tagger("La hundo")
    assert doc[1].pos_ == "NOUN" and doc[1].lemma_ == ""


def test_add_remove_pipe(nlp):
    """Verifica ``remove_pipe``, ``add_pipe`` y ``KeyError`` ante componentes inexistentes."""
    nlp2 = esperanto_nlp.load()
    removed = nlp2.remove_pipe("stopwords")
    assert nlp2.pipe_names == ["tagger", "lemmatizer"]
    nlp2.add_pipe(removed)
    assert nlp2.pipe_names[-1] == "stopwords"
    with pytest.raises(KeyError):
        nlp2.remove_pipe("nope")


def test_stopwords_api(tmp_path):
    """Verifica ``add``, ``remove``, listas personalizadas y carga ``from_file`` (JSON y TXT)."""
    sw = StopWords()
    assert "kaj" in sw and "Cxu" in sw and "hundo" not in sw
    sw.add("Hundo"); assert "hundo" in sw
    sw.remove("la"); assert "la" not in sw
    custom = esperanto_nlp.load(stopwords=StopWords(["hundo"]))
    assert custom("La hundo")[1].is_stop and not custom("La hundo")[0].is_stop
    f = tmp_path / "sw.json"; f.write_text(json.dumps(["foo", "bar"]), encoding="utf-8")
    assert len(StopWords.from_file(f)) == 2
    g = tmp_path / "sw.txt"; g.write_text("foo\nbar\nbaz\n", encoding="utf-8")
    assert "baz" in StopWords.from_file(g)


def test_realistic_paragraph(nlp):
    """Verifica el comportamiento integral del pipeline sobre un párrafo realista en Esperanto."""
    text = ("La kurantaj knaboj vidis la elefanton en Parizo, kaj ili ne scias, kio okazis. "
            "Ĝis la 15-a de majo ili restos tie.")
    doc = nlp(text)
    by = {t.text: t for t in doc}
    assert by["kurantaj"].lemma_ == "kuri" and by["elefanton"].lemma_ == "elefanto"
    assert by["Parizo"].pos_ == "PROPN" and by["kio"].pos_ == "PRON" and by["ne"].pos_ == "PART"
    assert len(list(doc.sents)) == 2
