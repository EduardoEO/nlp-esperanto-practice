"""Gestión de stop-words (palabras vacías).

La lista por defecto **no sale de ningún corpus**: se deriva de la gramática (palabras
cerradas del ``lexicon``: artículo, preposiciones, conjunciones, pronombres,
correlativos...) más las formas de *esti*. Así es independiente de la temática.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Set, Union

from .containers import Doc
from .lexicon import DEFAULT_STOPWORDS
from .normalization import normalize_word


class StopWords:
    """Componente del pipeline que marca ``token.is_stop`` según una lista normalizada.

    Attributes:
        name: Nombre identificador del componente en el pipeline (``"stopwords"``).
        words: Conjunto de palabras vacías normalizadas (NFC, sistema-x, minúsculas).
    """

    name = "stopwords"

    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        """Inicializa el filtro de stop-words.

        Args:
            words: Colección opcional de palabras personalizadas. Si es ``None``,
                utiliza ``DEFAULT_STOPWORDS`` derivado de ``lexicon.py``.
        """
        source = DEFAULT_STOPWORDS if words is None else words
        self.words: Set[str] = {normalize_word(w) for w in source}

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "StopWords":
        """Carga una lista de stop-words desde un archivo JSON o de texto plano.

        Admite un array JSON de cadenas o un archivo con una palabra por línea.

        Args:
            path: Ruta al archivo de stop-words (p. ej. ``resources/stopwords.json``).

        Returns:
            StopWords: Nueva instancia cargada con las palabras del archivo.

        Raises:
            ValueError: Si el archivo es JSON válido pero no contiene una lista.
        """
        raw = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("El JSON debe ser una lista de palabras")
            return cls(str(w) for w in data)
        except json.JSONDecodeError:
            return cls(line.strip() for line in raw.splitlines() if line.strip())

    def add(self, *words: str) -> None:
        """Añade una o más palabras al conjunto normalizándolas previamente.

        Args:
            *words: Palabras a incorporar como stop-words.
        """
        self.words.update(normalize_word(w) for w in words)

    def remove(self, *words: str) -> None:
        """Elimina una o más palabras del conjunto si están presentes.

        Args:
            *words: Palabras a retirar de la lista de stop-words.
        """
        for w in words:
            self.words.discard(normalize_word(w))

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and normalize_word(word) in self.words

    def __len__(self) -> int:
        return len(self.words)

    def __call__(self, doc: Doc) -> Doc:
        """Marca el atributo ``is_stop`` en cada token del documento.

        Args:
            doc: Documento ``Doc`` previamente tokenizado.

        Returns:
            Doc: El mismo documento ``Doc`` con ``is_stop`` actualizado in situ.
        """
        for tok in doc:
            tok.is_stop = tok.kind == "word" and tok.norm_ in self.words
        return doc
