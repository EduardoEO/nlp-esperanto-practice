"""Orquestador del pipeline: ``EsperantoNLP`` (equivalente a ``spacy.Language``).

Patrón *pipeline*: el texto entra por el tokenizador (que crea el ``Doc``) y pasa por
una cadena de componentes; cada uno recibe el ``Doc``, lo enriquece y lo devuelve.

    texto -> Tokenizer -> Tagger -> Lemmatizer -> StopWords -> Doc completo
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, List, Optional

from .containers import Doc
from .lemmatizer import Lemmatizer
from .stopwords import StopWords
from .tagger import Tagger
from .tokenizer import Tokenizer


class EsperantoNLP:
    def __init__(
        self,
        *,
        x_system: bool = True,
        stopwords: Optional[StopWords] = None,
        components: Optional[List[Callable[[Doc], Doc]]] = None,
    ) -> None:
        self.tokenizer = Tokenizer(x_system=x_system)
        self.stopwords = stopwords if stopwords is not None else StopWords()
        self.components: List[Callable[[Doc], Doc]] = (
            list(components) if components is not None
            else [Tagger(), Lemmatizer(), self.stopwords]
        )

    @property
    def pipe_names(self) -> List[str]:
        return [getattr(c, "name", type(c).__name__) for c in self.components]

    def add_pipe(self, component: Callable[[Doc], Doc], *, first: bool = False) -> None:
        self.components.insert(0, component) if first else self.components.append(component)

    def remove_pipe(self, name: str) -> Callable[[Doc], Doc]:
        for i, comp in enumerate(self.components):
            if getattr(comp, "name", type(comp).__name__) == name:
                return self.components.pop(i)
        raise KeyError(f"No hay ningún componente llamado {name!r}: {self.pipe_names}")

    def __call__(self, text: str) -> Doc:
        doc = self.tokenizer(text)
        for component in self.components:
            doc = component(doc)
        return doc

    def pipe(self, texts: Iterable[str]) -> Iterator[Doc]:
        """Procesa muchos textos (generador)."""
        for text in texts:
            yield self(text)


def load(**kwargs) -> EsperantoNLP:  # type: ignore[no-untyped-def]
    """Crea el pipeline por defecto: ``nlp = esperanto_nlp.load()``."""
    return EsperantoNLP(**kwargs)
