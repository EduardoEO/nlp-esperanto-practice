"""Analizador morfológico (módulo interno): el corazón de la librería.

``analyze(palabra)`` devuelve ``(raíz, POS, lema, rasgos)``. El *Tagger* y el
*Lemmatizer* son clases finas que solo llaman a esta función; así las reglas del
Esperanto viven en **un único sitio**. Los afijos (mal-, -ej-, -ist-...) se separan en ``_derivation.py`` (Opción B) y
se devuelven en ``base`` y ``affixes``.

Orden de decisión (la primera regla que encaja gana):

1. Números (``1950``) y signos.
2. Léxico de palabras cerradas (``kio``, ``tre``, ``ĉe``...).  <- antes que las terminaciones
3. Letras que no son del Esperanto (w, y, q, x...)  -> nombre propio o ``X``.
4. Siglas en mayúsculas cortas (``ONU``, ``UEFA``) -> ``PROPN``.
5. Numerales compuestos (``dudek``) y ordinales (``dudeka``).
6. Mayúscula inicial en mitad de frase -> ``PROPN``.
7. Terminaciones regulares: participios, verbos, -o, -a, -e.
8. Si nada encaja: ``PROPN`` (si empieza en mayúscula) o ``X``.

Limitaciones conocidas (sin diccionario de raíces es imposible evitarlas del todo):
palabras como ``elefanto`` parecen participios (``elef-ant-o``). Se mitigan con una
lista de excepciones y exigiendo raíces con vocal y ≥2 letras (``kato``, ``dato``).
"""
from __future__ import annotations

import re
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Tuple

from ._derivation import decompose
from .lexicon import LEXICON, is_compound_numeral
from .normalization import ESPERANTO_ALPHABET, VOWELS, normalize_word


class MorphAnalysis(NamedTuple):
    """Resultado del análisis: ``(root, pos, lemma, feats, base, affixes)``.

    * ``root``: raíz sin terminaciones gramaticales (``hundojn -> hund``).
    * ``base``: raíz sin **afijos derivativos** (``lernejo -> lern``) -> Opción B.
    * ``affixes``: afijos separados, p. ej. ``("mal-", "-ig")``.
    """

    root: str
    pos: str
    lemma: str
    feats: Mapping[str, str]
    base: str = ""
    affixes: Tuple[str, ...] = ()


def _mk(root: str, pos: str, lemma: str, feats: Optional[Mapping[str, str]] = None,
        *, derive: bool = False) -> MorphAnalysis:
    base, affixes = root, ()
    if derive:
        pre, base, suf = decompose(root)
        affixes = tuple(p + "-" for p in pre) + tuple("-" + x for x in suf)
    return MorphAnalysis(root, pos, lemma, MappingProxyType(dict(feats or {})), base, affixes)


_PARTICIPLE_RE = re.compile(r"(?P<stem>.+?)(?P<suf>ant|int|ont|at|it|ot)(?P<end>[oae])(?P<pl>j?)(?P<acc>n?)")
_VERB_RE = re.compile(r"(?P<stem>.+?)(?P<end>as|is|os|us|u|i)")
_NOUN_RE = re.compile(r"(?P<stem>.+?)o(?P<pl>j?)(?P<acc>n?)")
_ADJ_RE = re.compile(r"(?P<stem>.+?)a(?P<pl>j?)(?P<acc>n?)")
_ADV_RE = re.compile(r"(?P<stem>.+?)e(?P<acc>n?)")
_ORDINAL_RE = re.compile(r"(?P<stem>.+?)a(?P<pl>j?)(?P<acc>n?)")

_VERB_FEATS = {
    "as": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"},
    "is": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"},
    "os": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"},
    "us": {"VerbForm": "Fin", "Mood": "Cnd"},
    "u": {"VerbForm": "Fin", "Mood": "Imp"},
    "i": {"VerbForm": "Inf"},
}
_PART_TENSE = {"a": "Pres", "i": "Past", "o": "Fut"}   # primera letra del sufijo (ant/int/ont)
_PART_END_POS = {"o": "NOUN", "a": "ADJ", "e": "ADV"}

#: Raíces que NO son verbos aunque parezcan participios (elef-ant-o, kandid-at-o...).
_NON_PARTICIPLE_STEMS = frozenset({
    "elef", "gig", "diam", "orb", "reg", "spir", "kabr", "rab", "kandid", "advok",
    "diplom", "demokr", "aristokr",
})


def _valid_stem(stem: str) -> bool:
    """Una raíz razonable tiene ≥2 letras y al menos una vocal (descarta k-at-o, ŝt-at-o)."""
    return len(stem) >= 2 and any(c in VOWELS for c in stem)


def _number_feats(pl: str, acc: str) -> dict:
    feats = {}
    if pl:
        feats["Number"] = "Plur"
    if acc:
        feats["Case"] = "Acc"
    return feats


def _regular(norm: str) -> Optional[MorphAnalysis]:
    """Aplica las terminaciones regulares. ``None`` si no encaja ninguna."""
    # 1) Participios: -ant-/-int-/-ont- (activos), -at-/-it-/-ot- (pasivos) + o/a/e.
    m = _PARTICIPLE_RE.fullmatch(norm)
    if m and _valid_stem(m["stem"]) and m["stem"] not in _NON_PARTICIPLE_STEMS \
            and not (m["end"] == "e" and m["pl"]):
        suf = m["suf"]
        # Activos: ant/int/ont. Pasivos: at/it/ot. El tiempo lo da la primera letra.
        feats = {"VerbForm": "Part", "Tense": _PART_TENSE[suf[0]],
                 "Voice": "Act" if suf in ("ant", "int", "ont") else "Pass"}
        feats.update(_number_feats(m["pl"], m["acc"]))
        return _mk(m["stem"], _PART_END_POS[m["end"]], m["stem"] + "i", feats, derive=True)
    # 2) Verbos.
    m = _VERB_RE.fullmatch(norm)
    if m and _valid_stem(m["stem"]):
        pos = "AUX" if m["stem"] == "est" else "VERB"
        return _mk(m["stem"], pos, m["stem"] + "i", _VERB_FEATS[m["end"]], derive=True)
    # 3) Sustantivos, adjetivos, adverbios.
    m = _NOUN_RE.fullmatch(norm)
    if m and _valid_stem(m["stem"]):
        return _mk(m["stem"], "NOUN", m["stem"] + "o", _number_feats(m["pl"], m["acc"]), derive=True)
    m = _ADJ_RE.fullmatch(norm)
    if m and _valid_stem(m["stem"]):
        return _mk(m["stem"], "ADJ", m["stem"] + "a", _number_feats(m["pl"], m["acc"]), derive=True)
    m = _ADV_RE.fullmatch(norm)
    if m and _valid_stem(m["stem"]):
        return _mk(m["stem"], "ADV", m["stem"] + "e", _number_feats("", m["acc"]), derive=True)
    return None


def _proper_lemma(text: str) -> str:
    """Lema de un nombre propio: quita -j/-n de los nombres en -o (Parizon -> Parizo)."""
    if text.endswith("ojn"):
        return text[:-2]
    if text.endswith(("oj", "on")):
        return text[:-1]
    return text


def _propn(text: str) -> MorphAnalysis:
    feats = {}
    if text.endswith("ojn"):
        feats = {"Number": "Plur", "Case": "Acc"}
    elif text.endswith("oj"):
        feats = {"Number": "Plur"}
    elif text.endswith("on"):
        feats = {"Case": "Acc"}
    lemma = _proper_lemma(text)
    # La raíz va en minúscula y sin la -o final (Parizo -> pariz), igual que el resto de
    # palabras: así "Parizon" (PROPN) y "parizo" (a principio de frase) comparten raíz.
    stem = lemma[:-1] if lemma.endswith("o") and len(lemma) > 3 else lemma
    return _mk(stem.lower(), "PROPN", lemma, feats)


@lru_cache(maxsize=200_000)
def _analyze(word: str, sentence_start: bool) -> MorphAnalysis:
    norm = normalize_word(word)
    if not norm:
        return _mk("", "X", "")
    # 1) números y signos
    if norm[0].isdigit():
        return _mk(norm, "NUM", norm)
    if not any(c.isalpha() for c in norm):
        return _mk(norm, "PUNCT", word)

    # "Mayúscula inicial" real: un titular en MAYÚSCULAS largo no cuenta como nombre propio.
    capitalized = word[:1].isupper() and not (word.isupper() and len(word) > 4)

    # 2) palabras cerradas
    entry = LEXICON.get(norm)
    if entry is not None:
        return _mk(entry.lemma, entry.pos, entry.lemma, entry.feats)

    # 3) letras ajenas al Esperanto (w, y, q...)
    if any(c.isalpha() and c not in ESPERANTO_ALPHABET for c in norm):
        return _propn(word) if capitalized else _mk(norm, "X", norm)

    # 4) siglas cortas
    letters = word.replace("-", "")
    if word.isupper() and 2 <= len(letters) <= 4:
        return _mk(word, "PROPN", word)

    # 5) numerales compuestos y ordinales
    base = norm[:-1] if norm.endswith("n") else norm
    if is_compound_numeral(base):
        return _mk(base, "NUM", base, {"Case": "Acc"} if base != norm else {})
    m = _ORDINAL_RE.fullmatch(norm)
    if m and is_compound_numeral(m["stem"]):
        feats = {"NumType": "Ord", **_number_feats(m["pl"], m["acc"])}
        return _mk(m["stem"], "ADJ", m["stem"] + "a", feats)

    # 6) mayúscula en mitad de frase -> nombre propio
    if capitalized and not sentence_start:
        return _propn(word)

    # 7) terminaciones regulares
    result = _regular(norm)
    if result is not None:
        return result

    # 8) desconocida
    return _propn(word) if capitalized else _mk(norm, "X", norm)


def analyze(word: str, *, sentence_start: bool = False) -> MorphAnalysis:
    """Analiza una palabra suelta.

    Args:
        word: la palabra tal cual aparece en el texto (conservar mayúsculas es
            importante: distinguen nombres propios).
        sentence_start: ``True`` si es la primera palabra de una frase (entonces la
            mayúscula no indica nombre propio).

    >>> a = analyze("malbonigoj")
    >>> (a.root, a.base, a.lemma, a.affixes)
    ('malbonig', 'bon', 'malbonigo', ('mal-', '-ig'))
    """
    return _analyze(word, bool(sentence_start))


analyze.cache_clear = _analyze.cache_clear  # type: ignore[attr-defined]
