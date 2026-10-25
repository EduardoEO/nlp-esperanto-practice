"""Limpieza de párrafos (solo biblioteca estándar).

Vive en la librería —y no en el script del dataset— para que **entrenamiento e
inferencia limpien exactamente igual** (cuaderno y web scraping usan lo mismo que el
ETL). Si limpiasen distinto, el modelo vería datos diferentes a los de entrenamiento
(*train/serve skew*).
"""
from __future__ import annotations

import html
import re
import unicodedata

_TAGS = re.compile(r"<[^>]{0,200}>")
_TEMPLATES = re.compile(r"\{\{[^{}]{0,200}\}\}")
_URLS = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
_BRACKETS = re.compile(r"\[[^\[\]]{0,60}\]")            # [1], [nota 2], [citaĵo bezonata]
_EMPTY_PARENS = re.compile(r"\(\s*[,:–-]?\s*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_WS = re.compile(r"\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_TERMINAL = ('.', '!', '?', '…', '"', "'", '»', '”', ')', '’')


def clean_paragraph(text: str) -> str:
    """Quita HTML, plantillas, enlaces y referencias; elimina ``;`` y espacios sobrantes."""
    text = html.unescape(text)
    text = _TAGS.sub(" ", text)
    text = _TEMPLATES.sub(" ", text)
    text = _URLS.sub(" ", text)
    text = _BRACKETS.sub("", text)
    text = text.replace(";", ",")            # el separador del CSV no puede aparecer en el texto
    text = text.translate(_ZERO_WIDTH)
    text = _EMPTY_PARENS.sub(" ", text)
    text = _WS.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def is_valid_paragraph(
    text: str,
    *,
    min_chars: int = 120,
    max_chars: int = 2000,
    min_words: int = 15,
    min_alpha_ratio: float = 0.6,
    require_terminal_punct: bool = True,
) -> bool:
    """Filtro de calidad: descarta titulares, listas, restos de tablas y párrafos mínimos."""
    if not (min_chars <= len(text) <= max_chars):
        return False
    if len(text.split()) < min_words:
        return False
    if sum(c.isalpha() for c in text) / len(text) < min_alpha_ratio:
        return False
    if require_terminal_punct and not text.endswith(_TERMINAL):
        return False
    return True
