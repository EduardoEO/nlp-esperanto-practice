"""Puente con scikit-learn: usa ``EsperantoNLP`` como paso de un ``Pipeline``.

Se importa a mano (``from esperanto_nlp.sklearn_wrapper import EsperantoPreprocessor``)
para que la librería no obligue a instalar scikit-learn.

Decisiones de diseño:
* Hereda de ``BaseEstimator`` y ``TransformerMixin``: así funciona con ``clone``,
  ``GridSearchCV`` y ``cross_val_score``.
* No guarda el objeto ``nlp`` en el estimador (sería pesado de serializar): usa uno
  global con ``lru_cache``.
* El preprocesado no tiene estado (no aprende nada de los datos), por tanto cachearlo
  no produce *data leakage* y evita recalcularlo en cada fold de validación cruzada.
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
    """Transforma un texto con filtros deterministas y almacena el resultado en caché."""
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
            o ``"root"`` (sin afijos derivativos, Opción B).
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
        """Valida la representación elegida y devuelve el estimador sin aprender de los datos."""
        if self.feature not in _FEATURES:
            raise ValueError(f"feature debe ser uno de {sorted(_FEATURES)}, no {self.feature!r}")
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        """Devuelve un texto preprocesado por documento con la representación y filtros configurados."""
        attr = _FEATURES[self.feature]
        keep = tuple(self.keep_pos) if self.keep_pos is not None else None
        return [
            _preprocess(str(t), attr, self.drop_stop, self.drop_punct, self.drop_num,
                        self.lowercase, keep, self.min_len, self.x_system)
            for t in X
        ]
