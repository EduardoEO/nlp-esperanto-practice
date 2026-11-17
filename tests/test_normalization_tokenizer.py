import pytest

from esperanto_nlp.normalization import normalize_text, normalize_word
from esperanto_nlp.tokenizer import Tokenizer


@pytest.mark.parametrize("raw, expected", [
    ("cxu", "ĉu"), ("Cxu", "Ĉu"), ("gxi", "ĝi"), ("hxoro", "ĥoro"), ("jxus", "ĵus"),
    ("sxi", "ŝi"), ("aux", "aŭ"), ("Aux", "Aŭ"), ("eux", "eŭ"),
    ("lux", "lux"),            # "ux" tras consonante NO es ŭ (palabra extranjera)
    ("Luxemburgo", "Luxemburgo"),
])
def test_x_system(raw, expected):
    assert normalize_text(raw) == expected


def test_x_system_can_be_disabled():
    assert normalize_text("cxu", x_system=False) == "cxu"


def test_nfc_combining_accent():
    decomposed = "c\u0302u"                      # c + acento circunflejo combinante
    assert normalize_text(decomposed) == "ĉu"
    assert normalize_word("Ĉu") == "ĉu"


def toks(text):
    return [t.text for t in Tokenizer().tokenize(text)]


def test_basic_tokens():
    assert toks("La hundo kuras.") == ["La", "hundo", "kuras", "."]


def test_esperanto_letters_and_combining_are_single_tokens():
    assert toks("ĉiŭ ŝi") == ["ĉiŭ", "ŝi"]
    assert toks("c\u0302u") == ["c\u0302u"]


def test_numbers_hyphen_urls():
    assert toks("En 1.000,5 kaj 1950-a") == ["En", "1.000,5", "kaj", "1950-a"]
    assert toks("e-poŝto") == ["e-poŝto"]
    assert toks("Vidu https://eo.wikipedia.org/wiki/Esperanto.") == [
        "Vidu", "https://eo.wikipedia.org/wiki/Esperanto", "."]


def test_offsets_point_to_original_text():
    text = "Cxu  vi, ŝi?"
    for t in Tokenizer().tokenize(text):
        assert text[t.idx:t.end_idx] == t.text
    assert Tokenizer().tokenize(text)[0].norm_ == "ĉu"


def test_token_kinds():
    kinds = [t.kind for t in Tokenizer().tokenize("Mi vidis 3 hundojn: www.ex.org!")]
    assert kinds == ["word", "word", "number", "word", "punct", "url", "punct"]


@pytest.mark.parametrize("text, n_sents", [
    ("Mi venis. Vi iris. Ŝi restis.", 3),
    ("Ĉu vi venos? Jes! Bone.", 3),
    ("L. L. Zamenhof naskiĝis en 1859.", 1),     # iniciales no cortan
    ("Mi havas pomojn, bananojn k.t.p. kaj multe pli.", 1),
    ("Li diris: «Venu!» Poste foriris.", 2),
])
def test_sentence_segmentation(text, n_sents):
    from esperanto_nlp import load
    assert len(list(load()(text).sents)) == n_sents


def test_tokenizer_rejects_non_str():
    with pytest.raises(TypeError):
        Tokenizer()(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("text, expected", [
    ("Vidu ekz. la libron.", ["Vidu", "ekz.", "la", "libron", "."]),
    ("d-ro Zamenhof kaj S-ro Paris", ["d-ro", "Zamenhof", "kaj", "S-ro", "Paris"]),
    ("hundoj, katoj k.t.p. Poste", ["hundoj", ",", "katoj", "k.t.p.", "Poste"]),
    ("Skribu al a.b@ex-ample.org!", ["Skribu", "al", "a.b@ex-ample.org", "!"]),
])
def test_abbreviations_and_emails_are_single_tokens(text, expected):
    from esperanto_nlp import load
    assert [t.text for t in load()(text)] == expected


def test_abbreviation_kind_pos_and_enumeration_sentence_start():
    from esperanto_nlp import load
    doc = load()("Hundoj, katoj k.t.p. Poste venis la lupo.")
    abbr = [t for t in doc if t.kind == "abbr"][0]
    assert (abbr.pos_, abbr.lemma_) == ("X", "k.t.p.")
    assert [t for t in doc if t.text == "Poste"][0].is_sent_start
