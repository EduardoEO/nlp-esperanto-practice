"""Normalización de texto en Esperanto.

Dos problemas típicos del Esperanto en el ordenador:

1. **Letras con acento (ĉ ĝ ĥ ĵ ŝ ŭ).** Unicode permite escribir "ĉ" de dos formas:
   como un único carácter (U+0109) o como "c" + un acento combinado (U+0063 U+0302).
   Se ven igual, pero para Python son cadenas distintas. La *forma NFC* de Unicode
   obliga a usar siempre la versión de un solo carácter.
2. **Sistema-x.** Mucha gente sin teclado esperantista escribe "cx" en lugar de "ĉ"
   (``cx gx hx jx sx ux``). Lo convertimos a la letra real.

Cuidado con el sistema-x: "ux" no se puede convertir siempre, porque aparece en
palabras extranjeras (*lux*, *Luxemburgo*). Solo lo convertimos tras una vocal
(``aux -> aŭ``, ``eux -> eŭ``), que es donde el Esperanto usa "ŭ".
"""
from __future__ import annotations

import re
import unicodedata

#: Alfabeto esperantista (28 letras, sin q, w, x, y).
ESPERANTO_ALPHABET = frozenset("abcĉdefgĝhĥijĵklmnoprsŝtuŭvz")
VOWELS = frozenset("aeiou")

_X_TO_ACCENT = {"c": "ĉ", "g": "ĝ", "h": "ĥ", "j": "ĵ", "s": "ŝ"}
_X_RE = re.compile(r"([cghjsCGHJS])[xX]")
_UX_RE = re.compile(r"(?<=[aeiouAEIOU])([uU])[xX]")


def _x_replace(match: "re.Match[str]") -> str:
    base = match.group(1)
    accented = _X_TO_ACCENT[base.lower()]
    return accented.upper() if base.isupper() else accented


def _ux_replace(match: "re.Match[str]") -> str:
    return "Ŭ" if match.group(1).isupper() else "ŭ"


def normalize_text(text: str, *, x_system: bool = True) -> str:
    """Devuelve ``text`` en NFC y (opcionalmente) con el sistema-x convertido.

    >>> normalize_text("Cxu vi parolas Esperanton? Mi scias, ke jes.")
    'Ĉu vi parolas Esperanton? Mi scias, ke jes.'
    """
    text = unicodedata.normalize("NFC", text)
    if x_system:
        text = _X_RE.sub(_x_replace, text)
        text = _UX_RE.sub(_ux_replace, text)
    return text


def normalize_word(word: str, *, x_system: bool = True) -> str:
    """Normaliza una palabra y la pasa a minúsculas (``casefold``).

    Es la forma que usan el analizador morfológico y las stop-words para comparar.
    """
    return normalize_text(word, x_system=x_system).casefold()
