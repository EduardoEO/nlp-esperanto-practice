"""Etiquetador gramatical (POS-tagging) basado en reglas.

Es una clase muy fina: toda la lógica está en ``_morph_analyzer.analyze``.
Etiquetas: Universal POS (NOUN, VERB, AUX, ADJ, ADV, ADP, DET, PRON, CCONJ, SCONJ,
NUM, PART, INTJ, PROPN, PUNCT, X).
"""
from __future__ import annotations

from ._morph_analyzer import analyze
from .containers import Doc


class Tagger:
    """Componente del pipeline que asigna categoría POS, raíz flexiva y rasgos morfológicos.

    Attributes:
        name: Nombre identificador del componente en el pipeline (``"tagger"``).
    """

    name = "tagger"

    def __call__(self, doc: Doc) -> Doc:
        """Asigna ``pos_``, ``stem_`` y ``morph`` a cada token del documento.

        Args:
            doc: Documento ``Doc`` previamente tokenizado.

        Returns:
            Doc: El mismo documento ``Doc`` modificado in situ con las etiquetas asignadas.
        """
        for tok in doc:
            if tok.kind == "punct":
                tok.pos_, tok.stem_ = "PUNCT", tok.text
            elif tok.kind == "number":
                tok.pos_, tok.stem_ = "NUM", tok.text
            elif tok.kind == "url":
                tok.pos_, tok.stem_ = "X", tok.text
            elif tok.kind == "abbr":
                tok.pos_, tok.stem_, tok.morph = "X", tok.norm_, {"Abbr": "Yes"}
            else:
                a = analyze(tok.text, sentence_start=tok.is_sent_start)
                tok.pos_, tok.stem_, tok.morph = a.pos, a.root, dict(a.feats)
        return doc
