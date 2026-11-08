"""Léxico de palabras cerradas del Esperanto.

Las terminaciones (-o, -a, -e, -as...) resuelven casi todo el Esperanto, pero hay un
grupo pequeño de palabras que *no* las siguen: artículo, preposiciones, conjunciones,
pronombres, correlativos... Si no las tratásemos primero, ``kio`` (pronombre) parecería
un sustantivo por acabar en -o, y ``tre`` (adverbio) un adverbio con raíz "tr".

Por eso el analizador consulta primero este diccionario ``LEXICON``. Las formas con
-n/-j (``kiujn``, ``mian``) se **generan** aquí en lugar de escribirse a mano: el
Esperanto es regular y el código debe aprovecharlo.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, NamedTuple, Optional


class LexEntry(NamedTuple):
    pos: str
    lemma: str
    feats: Mapping[str, str]
    stop: bool


LEXICON: Dict[str, LexEntry] = {}


def _reg(form: str, pos: str, lemma: str, feats: Optional[dict] = None, stop: bool = False) -> None:
    if form in LEXICON:
        raise ValueError(f"Forma duplicada en el léxico: {form!r}")
    LEXICON[form] = LexEntry(pos, lemma, MappingProxyType(dict(feats or {})), stop)


def _reg_inflected(base: str, pos: str, lemma: str, feats: dict, stop: bool, inflections: List[str]) -> None:
    """Registra ``base`` y sus variantes con -n (acusativo) y/o -j (plural)."""
    for infl in inflections:
        f = dict(feats)
        if "j" in infl:
            f["Number"] = "Plur"
        if "n" in infl:
            f["Case"] = "Acc"
        _reg(base + infl, pos, lemma, f, stop)


# --- Artículo, preposiciones, conjunciones, partículas, interjecciones -------------
_reg("la", "DET", "la", {"Definite": "Def"}, True)

PREPOSITIONS = (
    "al anstataŭ antaŭ apud ĉe ĉirkaŭ da de dum ekde ekster el en ĝis inter je kontraŭ "
    "krom kun laŭ malantaŭ malgraŭ per plus minus po por post preter pri pro sen sub "
    "super sur tra trans"
).split()
for _w in PREPOSITIONS:
    _reg(_w, "ADP", _w, stop=True)

for _w in ("kaj", "aŭ", "sed", "nek", "do"):
    _reg(_w, "CCONJ", _w, stop=True)
for _w in ("ke", "se", "ĉar", "ol", "kvankam", "kvazaŭ"):
    _reg(_w, "SCONJ", _w, stop=True)
for _w in ("ne", "ĉu", "jen", "ĉi"):
    _reg(_w, "PART", _w, stop=True)
for _w in ("jes", "nu", "ho", "ha", "ve", "hura", "adiaŭ"):
    _reg(_w, "INTJ", _w)

# Adverbios que NO terminan en -e (y por tanto no los resuelve la regla general).
for _w in ("ankaŭ", "ankoraŭ", "jam", "nur", "eĉ", "ja", "tre", "tro", "pli", "plej",
           "malpli", "malplej", "nun", "tuj", "tamen", "ju", "des", "plu", "for"):
    _reg(_w, "ADV", _w, stop=True)
for _w in ("hodiaŭ", "hieraŭ", "morgaŭ", "baldaŭ", "preskaŭ", "apenaŭ", "almenaŭ", "antaŭhieraŭ"):
    _reg(_w, "ADV", _w)

_reg("ambaŭ", "DET", "ambaŭ", stop=True)

# --- Pronombres personales y posesivos -------------------------------------------
PERSONAL_PRONOUNS = ("mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si")
for _p in PERSONAL_PRONOUNS:
    _reg_inflected(_p, "PRON", _p, {"PronType": "Prs"}, True, ["", "n"])
    if _p != "oni":  # "onia" no se usa
        _reg_inflected(_p + "a", "DET", _p + "a", {"PronType": "Prs", "Poss": "Yes"}, True,
                       ["", "n", "j", "jn"])

# --- Correlativos: prefijo x sufijo (5 x 9 = 45 palabras, ¡generadas, no escritas!) ---
CORRELATIVE_PREFIXES = {"ki": "Int", "ti": "Dem", "i": "Ind", "ĉi": "Tot", "neni": "Neg"}
CORRELATIVE_SUFFIXES = [
    ("u", "PRON", ["", "n", "j", "jn"]),   # kiu, kiun, kiuj, kiujn
    ("o", "PRON", ["", "n"]),              # kio, kion
    ("a", "DET", ["", "n", "j", "jn"]),    # kia, kian, kiaj, kiajn
    ("e", "ADV", ["", "n"]),               # kie, kien (dirección)
    ("am", "ADV", [""]),                   # kiam
    ("el", "ADV", [""]),                   # kiel
    ("al", "ADV", [""]),                   # kial
    ("es", "DET", [""]),                   # kies
    ("om", "ADV", [""]),                   # kiom
]
for _pre, _ptype in CORRELATIVE_PREFIXES.items():
    for _suf, _pos, _infl in CORRELATIVE_SUFFIXES:
        _reg_inflected(_pre + _suf, _pos, _pre + _suf, {"PronType": _ptype}, True, _infl)

# --- Numerales cardinales -----------------------------------------------------
NUMERAL_ROOTS = ("nul", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ",
                 "dek", "cent", "mil")
for _r in NUMERAL_ROOTS:
    _reg_inflected(_r, "NUM", _r, {}, False, ["", "n"])


def is_compound_numeral(word: str) -> bool:
    """``True`` si ``word`` se forma solo con raíces numerales: ``dudek``, ``tricent``.

    Programación dinámica: ``ok[i]`` es cierto si el prefijo de longitud ``i`` se puede
    partir en raíces numerales.
    """
    if not word:
        return False
    ok = [True] + [False] * len(word)
    for i in range(1, len(word) + 1):
        for root in NUMERAL_ROOTS:
            start = i - len(root)
            if start >= 0 and ok[start] and word[start:i] == root:
                ok[i] = True
                break
    return ok[-1]


#: Formas de "esti" (verbo copulativo, se etiqueta AUX). También son stop-words.
ESTI_FORMS = frozenset("est" + e for e in ("i", "as", "is", "os", "us", "u"))

#: Stop-words por defecto: todo lo marcado con stop=True + formas de "esti".
DEFAULT_STOPWORDS = frozenset(f for f, e in LEXICON.items() if e.stop) | ESTI_FORMS
