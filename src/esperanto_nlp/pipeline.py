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
    """Orquestador secuencial del pipeline lingüístico para Esperanto.

    Attributes:
        tokenizer: Instancia de ``Tokenizer`` que segmenta el texto en ``Doc``.
        stopwords: Instancia de ``StopWords`` por defecto para el pipeline.
        components: Lista ordenada de componentes ejecutados tras el tokenizador.
    """

    def __init__(
        self,
        *,
        x_system: bool = True,
        stopwords: Optional[StopWords] = None,
        components: Optional[List[Callable[[Doc], Doc]]] = None,
    ) -> None:
        """Inicializa el pipeline con sus componentes lingüísticos.

        Args:
            x_system: Si es ``True``, convierte dígrafos del sistema-x en la normalización.
            stopwords: Filtro personalizado de stop-words. Si es ``None``, usa ``StopWords()``.
            components: Lista opcional de componentes que reemplaza a ``[Tagger(), Lemmatizer(), stopwords]``.
        """
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
        """Inserta un componente nuevo en el pipeline.

        Args:
            component: Callable que recibe un ``Doc`` y devuelve el ``Doc`` enriquecido.
            first: Si es ``True``, lo inserta al principio de ``components``; en caso
                contrario, lo añade al final.
        """
        self.components.insert(0, component) if first else self.components.append(component)

    def remove_pipe(self, name: str) -> Callable[[Doc], Doc]:
        """Elimina y devuelve el componente cuyo nombre coincide con ``name``.

        Args:
            name: Identificador del componente (atributo ``name`` o nombre de su clase).

        Returns:
            Callable[[Doc], Doc]: El componente extraído del pipeline.

        Raises:
            KeyError: Si no existe ningún componente registrado con ese nombre.
        """
        for i, comp in enumerate(self.components):
            if getattr(comp, "name", type(comp).__name__) == name:
                return self.components.pop(i)
        raise KeyError(f"No hay ningún componente llamado {name!r}: {self.pipe_names}")

    def __call__(self, text: str) -> Doc:
        """Procesa un texto individual pasando por el tokenizador y los componentes.

        Args:
            text: Texto de entrada en Esperanto.

        Returns:
            Doc: Documento anotado con POS, lemas, raíces, afijos y stop-words.
        """
        doc = self.tokenizer(text)
        for component in self.components:
            doc = component(doc)
        return doc

    def pipe(self, texts: Iterable[str]) -> Iterator[Doc]:
        """Procesa una secuencia de textos de forma perezosa mediante un generador.

        Args:
            texts: Iterable de cadenas de entrada.

        Yields:
            Doc: Cada documento procesado secuencialmente.
        """
        for text in texts:
            yield self(text)


def load(**kwargs) -> EsperantoNLP:  # type: ignore[no-untyped-def]
    """Crea e inicializa el pipeline por defecto: ``nlp = esperanto_nlp.load()``.

    Args:
        **kwargs: Argumentos de configuración reenviados a ``EsperantoNLP``.

    Returns:
        EsperantoNLP: Instancia lista para analizar textos.
    """
    return EsperantoNLP(**kwargs)
