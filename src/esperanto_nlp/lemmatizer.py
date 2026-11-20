"""Lematizador basado en reglas.

Lema = raíz + terminación canónica de su categoría
(sustantivo ``-o``, adjetivo ``-a``, adverbio ``-e``, verbo ``-i``).
Además rellena ``root_`` y ``affixes`` (raíz primaria sin afijos derivativos).
Los participios se lematizan al **infinitivo** (``kuranta -> kuri``), decisión de
diseño que reduce vocabulario; está documentada en ``docs/decisiones-diseno.md``.
"""
from __future__ import annotations

from ._morph_analyzer import analyze
from .containers import Doc


class Lemmatizer:
    """Componente del pipeline que asigna lemas canónicos, raíces primarias y afijos.

    Attributes:
        name: Nombre identificador del componente en el pipeline (``"lemmatizer"``).
    """

    name = "lemmatizer"

    def __call__(self, doc: Doc) -> Doc:
        """Rellena ``lemma_``, ``root_`` y ``affixes`` en cada token del documento.

        Args:
            doc: Documento ``Doc`` previamente tokenizado.

        Returns:
            Doc: El mismo documento ``Doc`` modificado in situ.
        """
        for tok in doc:
            if tok.kind == "word":
                a = analyze(tok.text, sentence_start=tok.is_sent_start)
                tok.lemma_, tok.root_, tok.affixes = a.lemma, a.base, a.affixes
            elif tok.kind == "abbr":
                tok.lemma_ = tok.root_ = tok.norm_
            else:
                tok.lemma_ = tok.root_ = tok.text
        return doc
