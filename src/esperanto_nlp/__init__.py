"""esperanto_nlp: librería de NLP para Esperanto basada en reglas, con API estilo spaCy.

    >>> import esperanto_nlp
    >>> nlp = esperanto_nlp.load()
    >>> doc = nlp("La hundoj kuras rapide.")
    >>> [(t.text, t.pos_, t.lemma_) for t in doc][:3]
    [('La', 'DET', 'la'), ('hundoj', 'NOUN', 'hundo'), ('kuras', 'VERB', 'kuri')]
"""
from ._morph_analyzer import MorphAnalysis, analyze
from .containers import Doc, Span, Token
from .pipeline import EsperantoNLP, load

__version__ = "0.3.1"
__all__ = ["load", "EsperantoNLP", "Doc", "Span", "Token", "analyze", "MorphAnalysis", "__version__"]
