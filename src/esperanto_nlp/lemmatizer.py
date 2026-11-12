"""Lematizador basado en reglas.

Lema = raíz + terminación canónica de su categoría
(sustantivo ``-o``, adjetivo ``-a``, adverbio ``-e``, verbo ``-i``).
Además rellena ``root_`` y ``affixes`` (Opción B: raíz sin afijos derivativos).
Los participios se lematizan al **infinitivo** (``kuranta -> kuri``), decisión de
diseño que reduce vocabulario; está documentada en ``docs/decisiones-diseno.md``.
"""
from __future__ import annotations

from ._morph_analyzer import analyze
from .containers import Doc


class Lemmatizer:
    name = "lemmatizer"

    def __call__(self, doc: Doc) -> Doc:
        for tok in doc:
            if tok.kind == "word":
                a = analyze(tok.text, sentence_start=tok.is_sent_start)
                tok.lemma_, tok.root_, tok.affixes = a.lemma, a.base, a.affixes
            elif tok.kind == "abbr":
                tok.lemma_ = tok.root_ = tok.norm_
            else:
                tok.lemma_ = tok.root_ = tok.text
        return doc
