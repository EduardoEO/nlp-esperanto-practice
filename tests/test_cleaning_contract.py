from esperanto_nlp.cleaning import clean_paragraph, is_valid_paragraph
from esperanto_nlp.contract import validate_csv_text, validate_file


def test_clean_paragraph():
    raw = ("La <b>hundo</b> kuras[1] kaj saltas; ĝi estas bona (; ) vidu https://ex.org/a "
           "kaj {{ŝablono}}[citaĵo bezonata] .\n  Fino &amp; fino.")
    out = clean_paragraph(raw)
    assert ";" not in out and "<" not in out and "http" not in out and "[" not in out
    assert "{{" not in out and "\n" not in out and "  " not in out and "()" not in out
    assert "&" in out and out.startswith("La hundo kuras kaj saltas, ĝi")
    assert "kaj ." not in out


def test_is_valid_paragraph():
    good = ("La futbala teamo gajnis la matĉon per tri goloj kontraŭ nul kaj la spektantoj "
            "aplaŭdis la ludantojn dum longa tempo post la fino.")
    assert is_valid_paragraph(good)
    assert not is_valid_paragraph("Tro mallonga.")
    assert not is_valid_paragraph(good[:-1])                       # sin punto final
    assert is_valid_paragraph(good[:-1], require_terminal_punct=False)
    assert not is_valid_paragraph("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28.")


def test_contract_ok_on_fixture(mini_dataset_path):
    rep = validate_file(mini_dataset_path, min_per_class=5)
    assert rep.ok, rep.summary()
    assert rep.stats["clases"] == {"futbalo": 6, "korbopilko": 6, "teniso": 6}


def test_contract_detects_violations():
    bad = ("text;class\n"
           "\"hola; adios\";a\n"
           "<p>html</p>;a\n"
           "ver http://x.org;b\n"
           "ref [1] aquí;b\n"
           "ok;\n"
           "una;sola;columna;mas\n")
    rep = validate_csv_text(bad, min_per_class=1)
    joined = "\n".join(rep.errors)
    for needle in ("contiene ';'", "HTML", "enlaces", "referencias", "clase vacía", "columnas"):
        assert needle in joined, needle
    assert not rep.ok


def test_contract_header_and_encoding(tmp_path):
    assert not validate_csv_text("texto;clase\nx;y\n").ok
    assert not validate_csv_text("\ufefftext;class\nx;y\n").ok
    p = tmp_path / "latin.csv"
    p.write_bytes(b"text;class\ncaf\xe9;a\n")      # \xe9 solo, sin continuación: UTF-8 inválido
    assert not validate_file(p).ok
