"""Tokenizador basado en expresiones regulares (Opción A) + marcado de frases.

Idea clave: **no modificamos el texto original**. Cada ``Token`` guarda su ``text``
tal cual y su posición (``idx``). La versión normalizada (sistema-x, NFC, minúsculas)
va aparte, en ``token.norm_``. Así los desplazamientos nunca se descuadran.

La regex prueba, en orden, cinco patrones (el orden importa: gana el primero):
URL/correo -> abreviatura -> número -> palabra -> puntuación.
"""
from __future__ import annotations

import re
from typing import List

from .containers import Doc, Token
from .normalization import normalize_word

# Carácter de palabra: letra/dígito Unicode o acento combinante (c + U+0302).
_WCH = r"(?:[^\W_]|[\u0300-\u036f])"
_URL = r"(?:https?://|www\.)[^\s<>\"]*[^\s<>\".,;:!?)\]'’]"
_EMAIL = rf"{_WCH}[\w.+-]*@[\w-]+(?:\.[\w-]+)+"
# Abreviaturas que se mantienen como UN token (con su punto): "ekz.", "k.t.p.", "d-ro"...
# Van de la más larga a la más corta para que "k.t.p." gane a "k.".
_ABBR_LIST = ("k.t.p.", "k.a.", "t.e.", "p.k.", "a.k.", "ekz.", "ktp.", "prof.", "inĝ.", "ĝen.",
              "vd.", "d-ino", "s-ino", "f-ino", "d-ro", "s-ro", "n-ro")
_ABBR = r"(?<![\w.-])(?i:" + "|".join(re.escape(a) for a in _ABBR_LIST) + r")(?![\w-])"
_NUMBER = rf"\d+(?:[.,:]\d+)*(?:-{_WCH}+)?"          # 1950, 1.000,5, 12:30, 20-a
_WORD = rf"[^\W\d_]{_WCH}*(?:[-'’]{_WCH}+)*"           # hundo, e-poŝto, l'ago
_PUNCT = r"\.{3}|…|[^\w\s]|_"

TOKEN_RE = re.compile(
    rf"(?P<url>{_URL}|{_EMAIL})|(?P<abbr>{_ABBR})|(?P<number>{_NUMBER})|(?P<word>{_WORD})|(?P<punct>{_PUNCT})"
)

_TERMINALS = {".", "!", "?", "…", "..."}
_CLOSERS = set("\"'»”’)]}")
_OPENERS = set("\"'«“„(¿¡[{")
#: Abreviaturas frecuentes: un punto tras ellas NO termina la frase.
_LIST_ENDERS = frozenset({"ktp.", "k.t.p."})
_ABBREVIATIONS = frozenset({"ekz", "vd", "ktp", "ks", "kp", "pp", "ĝen", "fr", "nr", "p"})


def _period_ends_sentence(tokens: List[Token], i: int) -> bool:
    """Decide si el punto ``tokens[i]`` cierra una frase."""
    if i == 0:
        return False
    prev = tokens[i - 1]
    if prev.kind == "word":
        # Iniciales ("L. L. Zamenhof") y siglas con puntos ("k.t.p.") -> letra suelta.
        if len(prev.norm_) == 1 and prev.norm_.isalpha():
            return False
        if prev.norm_ in _ABBREVIATIONS:
            return False
    return True


def mark_sentences(tokens: List[Token]) -> None:
    """Rellena ``is_sent_start`` con reglas simples.

    Una frase empieza en el primer token y tras ``. ! ? …`` siempre que el siguiente
    token no empiece en minúscula (``"k.t.p. kaj"`` no corta; ``"Venis. La..."`` sí).
    """
    if not tokens:
        return
    tokens[0].is_sent_start = True
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok.kind == "punct" and tok.text in _TERMINALS:
            if tok.text == "." and not _period_ends_sentence(tokens, i):
                i += 1
                continue
            j = i + 1
            while j < n and tokens[j].kind == "punct" and (
                tokens[j].text in _CLOSERS or tokens[j].text in _TERMINALS
            ):
                j += 1
            while j < n and tokens[j].kind == "punct" and tokens[j].text in _OPENERS:
                j += 1
            if j < n and not tokens[j].text[:1].islower():
                tokens[j].is_sent_start = True
            i = max(j, i + 1)
            continue
        # "ktp." / "k.t.p." cierran enumeraciones: si sigue una mayúscula, empieza frase.
        if tok.kind == "abbr" and tok.norm_ in _LIST_ENDERS and i + 1 < n \
                and tokens[i + 1].text[:1].isupper():
            tokens[i + 1].is_sent_start = True
        i += 1


class Tokenizer:
    """Convierte texto crudo en un ``Doc`` con ``Token`` inicializados."""

    name = "tokenizer"

    def __init__(self, *, x_system: bool = True) -> None:
        self.x_system = x_system

    def tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        for m in TOKEN_RE.finditer(text):
            kind = m.lastgroup or "punct"
            raw = m.group()
            if kind == "word":
                norm = normalize_word(raw, x_system=self.x_system)
            else:
                norm = raw.lower() if kind == "abbr" else raw
            tokens.append(Token(text=raw, idx=m.start(), kind=kind, norm_=norm))
        mark_sentences(tokens)
        return tokens

    def __call__(self, text: str) -> Doc:
        if not isinstance(text, str):
            raise TypeError(f"Se esperaba str, se recibió {type(text).__name__}")
        return Doc(text, self.tokenize(text))
