"""Adapta el análisis lingüístico de Esperanto al clasificador de textos.

El preprocesador recibe un iterable de documentos y devuelve una cadena por
documento. El cuaderno de clasificación cambia el parámetro ``feature`` para
comparar texto original, formas normalizadas, lemas, tallos y raíces con el
mismo vectorizador TF-IDF.

Uso:
    from esperanto_nlp.sklearn_wrapper import EsperantoPreprocessor
    prep = EsperantoPreprocessor(feature="lemma")
    prep.fit_transform(["La hundoj kuras."])  # ["hundo kuri"]

La caché LRU reutiliza el análisis del mismo texto con la misma configuración
entre particiones de validación cruzada. El análisis de cada documento es
independiente del corpus. El vocabulario y los pesos TF-IDF se aprenden en
el vectorizador. scikit-learn solo se requiere al importar este módulo.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

from sklearn.base import BaseEstimator, TransformerMixin

from .pipeline import EsperantoNLP, load

_FEATURES = {"text": "text", "norm": "norm_", "lemma": "lemma_", "stem": "stem_", "root": "root_"}


@lru_cache(maxsize=4)
def _get_nlp(x_system: bool) -> EsperantoNLP:
    """Reutiliza el pipeline asociado a la configuración del sistema-x."""
    return load(x_system=x_system)


@lru_cache(maxsize=200_000)
def _preprocess(
    text: str, attr: str, drop_stop: bool, drop_punct: bool, drop_num: bool,
    lowercase: bool, keep_pos: Optional[Tuple[str, ...]], min_len: int, x_system: bool,
) -> str:
    """Aplica los filtros del pipeline a un documento y cachea el resultado.

    La clave de caché incluye el texto, el atributo y todas las opciones.
    Los espacios internos de un token se sustituyen por guiones bajos para
    que el vectorizador lo trate como una sola unidad al usar ``str.split``.
    """
    doc = _get_nlp(x_system)(text)
    toks = doc.to_list(attr, drop_stop=drop_stop, drop_punct=drop_punct,
                       drop_num=drop_num, keep_pos=keep_pos)
    if lowercase:
        toks = [t.lower() for t in toks]
    return " ".join(t.replace(" ", "_") for t in toks if len(t) >= min_len)


class EsperantoPreprocessor(BaseEstimator, TransformerMixin):
    """Convierte textos en cadenas de "palabras" separadas por espacios.

    Args:
        feature: ``"text"`` (original), ``"norm"``, ``"lemma"``, ``"stem"`` (sin terminaciones)
            o ``"root"`` (raíz léxica tras separar los afijos derivativos).
        drop_stop: quitar stop-words.
        drop_punct: quitar signos de puntuación.
        drop_num: quitar números.
        lowercase: pasar a minúsculas.
        keep_pos (list or None): POS a conservar (``None`` = todos).
        min_len: longitud mínima de cada token.
        x_system: convertir sistema-x.

    Úsalo antes de ``TfidfVectorizer(tokenizer=str.split, token_pattern=None,
    lowercase=False)``.
    """

    def __init__(self, feature: str = "lemma", drop_stop: bool = True, drop_punct: bool = True,
                 drop_num: bool = True, lowercase: bool = True, keep_pos=None,
                 min_len: int = 2, x_system: bool = True) -> None:
        """Guarda los parámetros del preprocesador para permitir su clonado con scikit-learn."""
        self.feature = feature
        self.drop_stop = drop_stop
        self.drop_punct = drop_punct
        self.drop_num = drop_num
        self.lowercase = lowercase
        self.keep_pos = keep_pos
        self.min_len = min_len
        self.x_system = x_system

    def fit(self, X: Iterable[str], y=None) -> "EsperantoPreprocessor":  # noqa: ARG002
        """Valida la configuración antes de transformar el corpus.

        Args:
            X: Documentos recibidos a través de la interfaz de scikit-learn.
            y: Etiquetas opcionales. El preprocesador prescinde de ellas.

        Returns:
            El mismo estimador. La transformación depende de sus parámetros.

        Raises:
            ValueError: Si ``feature`` no es una representación admitida.
        """
        if self.feature not in _FEATURES:
            raise ValueError(f"feature debe ser uno de {sorted(_FEATURES)}, no {self.feature!r}")
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        """Genera las cadenas que consume el vectorizador TF-IDF.

        Args:
            X: Documentos en el orden en que se deben transformar.

        Returns:
            Una lista en el mismo orden, con los tokens seleccionados de
            cada documento separados por espacios.
        """
        attr = _FEATURES[self.feature]
        keep = tuple(self.keep_pos) if self.keep_pos is not None else None
        return [
            _preprocess(str(t), attr, self.drop_stop, self.drop_punct, self.drop_num,
                        self.lowercase, keep, self.min_len, self.x_system)
            for t in X
        ]
