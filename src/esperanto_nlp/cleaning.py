"""Módulo de limpieza, desinfección y filtrado de párrafos para Esperanto.

Este módulo implementa las transformaciones de texto puro necesarias para sanitizar
documentos extraídos de fuentes web (como Vikipedio o páginas en Esperanto) y asegurar
que cumplan con los estándares de calidad del corpus.

Diseño arquitectónico:
Las funciones de limpieza residen en la librería central (`esperanto_nlp`) en lugar
de aislarse en scripts de preparación de datos. Esto garantiza consistencia absoluta
entre la etapa de extracción (ETL), el entrenamiento de modelos de Machine Learning
y la inferencia en producción (web scraping), previniendo el sesgo entre entrenamiento
y servicio (*train/serve skew*).

Capacidades principales:
- Decodificación de entidades HTML y eliminación de etiquetas de marcado.
- Limpieza de sintaxis wiki (plantillas `{{...}}`, citas bibliográficas `[...]`).
- Supresión de URLs y enlaces externos.
- Preservación de la integridad del formato CSV (sustitución segura de `;` por `,`).
- Normalización Unicode NFC y eliminación de caracteres invisibles de ancho cero.
- Filtro de calidad sintáctica y volumétrica para descartar ruido (titulares,
  fragmentos incompletos o fórmulas numéricas).
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
    """Desinfecta y normaliza un fragmento textual plano en Esperanto.

    Aplica una secuencia determinista de filtros para eliminar artefactos web,
    etiquetas de marcado y caracteres conflictivos, garantizando que el texto
    sea seguro para su almacenamiento tabular y posterior tokenización.

    Operaciones realizadas:
        1. Decodifica entidades HTML (ej. `&amp;` -> `&`).
        2. Elimina etiquetas HTML y plantillas de MediaWiki (`{{...}}`).
        3. Elimina URLs completas y referencias bibliográficas entre corchetes (`[1]`).
        4. Reemplaza el delimitador reservado `;` por `,` para respetar el contrato CSV.
        5. Remueve caracteres invisibles de ancho cero (`\\u200b`, `\\ufeff`, etc.).
        6. Elimina paréntesis vacíos o con puntuación huérfana.
        7. Colapsa espacios y saltos de línea continuos en un único espacio en blanco.
        8. Normaliza caracteres Unicode en forma compuesta NFC.
        9. Ajusta la tipografía eliminando espacios espurios antes de signos de puntuación.

    Args:
        text: Cadena de texto sin procesar extraída de páginas web o artículos wiki.

    Returns:
        str: Texto limpio, normalizado y sin saltos de línea ni punto y coma.
    """
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
    """Aplica filtros de calidad volumétrica y sintáctica sobre un párrafo.

    Determina si un texto posee la suficiente densidad informativa y corrección
    estructural para ser incluido en el conjunto de entrenamiento de clasificación
    temática, descartando fragmentos residuales, titulares, listas o fórmulas.

    Criterios evaluados:
        - Rango de longitud: Debe tener entre `min_chars` y `max_chars` caracteres.
        - Volumen léxico: Debe contener al menos `min_words` palabras.
        - Densidad alfabética: La proporción de letras alfabéticas respecto al total
          de caracteres debe ser al menos `min_alpha_ratio` (descarta tablas y código).
        - Puntuación terminal: Si `require_terminal_punct` es True, debe finalizar
          en un signo de cierre válido (`.`, `!`, `?`, `…`, comillas o paréntesis).

    Args:
        text: Párrafo previamente limpiado con `clean_paragraph`.
        min_chars: Longitud mínima en caracteres (por defecto 120).
        max_chars: Longitud máxima en caracteres (por defecto 2000).
        min_words: Número mínimo de palabras separadas por espacios (por defecto 15).
        min_alpha_ratio: Proporción mínima de caracteres alfabéticos (por defecto 0.6).
        require_terminal_punct: Exigir signo ortográfico de cierre (por defecto True).

    Returns:
        bool: True si el párrafo supera todos los umbrales de calidad; False en caso contrario.
    """
    if not (min_chars <= len(text) <= max_chars):
        return False
    if len(text.split()) < min_words:
        return False
    if sum(c.isalpha() for c in text) / len(text) < min_alpha_ratio:
        return False
    if require_terminal_punct and not text.endswith(_TERMINAL):
        return False
    return True
