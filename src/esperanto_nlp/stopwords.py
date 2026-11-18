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
    name = "stopwords"

    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        source = DEFAULT_STOPWORDS if words is None else words
        self.words: Set[str] = {normalize_word(w) for w in source}

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "StopWords":
        """Carga una lista desde JSON (lista de cadenas) o texto (una palabra por línea)."""
        raw = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("El JSON debe ser una lista de palabras")
            return cls(str(w) for w in data)
        except json.JSONDecodeError:
            return cls(line.strip() for line in raw.splitlines() if line.strip())

    def add(self, *words: str) -> None:
        self.words.update(normalize_word(w) for w in words)

    def remove(self, *words: str) -> None:
        for w in words:
            self.words.discard(normalize_word(w))

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and normalize_word(word) in self.words

    def __len__(self) -> int:
        return len(self.words)

    def __call__(self, doc: Doc) -> Doc:
        for tok in doc:
            tok.is_stop = tok.kind == "word" and tok.norm_ in self.words
        return doc
