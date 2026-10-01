"""Librería de Procesamiento de Lenguaje Natural basada en reglas para Esperanto."""

from esperanto_nlp.containers import Doc, Token
from esperanto_nlp.lemmatizer import Lemmatizer
from esperanto_nlp.pipeline import EsperantoNLP, load
from esperanto_nlp.stopwords import StopwordManager
from esperanto_nlp.tagger import POSTagger
from esperanto_nlp.tokenizer import Tokenizer

__version__ = "0.1.0"
__all__ = [
    "load",
    "EsperantoNLP",
    "Doc",
    "Token",
    "Tokenizer",
    "POSTagger",
    "Lemmatizer",
    "StopwordManager",
]
