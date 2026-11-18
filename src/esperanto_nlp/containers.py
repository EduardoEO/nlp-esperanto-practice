"""Contenedores de datos: ``Token``, ``Span`` y ``Doc``.

Siguen la idea de spaCy: ``nlp(texto)`` devuelve un ``Doc``, que es una lista de
``Token``. Son *modelos anémicos*: no saben procesar texto, solo guardan atributos.
Quien rellena los atributos son los componentes del pipeline (tagger, lemmatizer...).

Convención de nombres (igual que spaCy): los atributos que terminan en ``_`` son
cadenas legibles (``token.pos_ == "NOUN"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .normalization import normalize_word


@dataclass(eq=False)
class Token:
    """Una unidad del texto (palabra, número, signo de puntuación o URL).

    Attributes:
        text: texto **original**, tal cual aparece en el documento.
        idx: posición del primer carácter de ``text`` en el texto original.
        kind: ``"word"``, ``"number"``, ``"punct"``, ``"abbr"`` o ``"url"`` (URLs y correos) (lo fija el tokenizador).
        norm_: forma normalizada (NFC, sistema-x convertido, minúsculas).
        lemma_: forma canónica (``hundojn -> hundo``, ``kuris -> kuri``).
        stem_: raíz sin terminaciones (``hundojn -> hund``).
        root_: raíz sin afijos derivativos (``lernejo -> lern``, Opción B).
        affixes: afijos separados (``("mal-", "-ig")``).
        pos_: categoría gramatical (etiquetas Universal POS: NOUN, VERB, ADJ...).
        morph: rasgos morfológicos, p. ej. ``{"Number": "Plur", "Case": "Acc"}``.
        is_sent_start: ``True`` si el token empieza una frase.
        is_stop: ``True`` si es una stop-word.

    Se compara por identidad (``eq=False``) para poder usar tokens como claves.
    """

    text: str
    idx: int = 0
    kind: str = "word"
    norm_: str = ""
    lemma_: str = ""
    stem_: str = ""
    root_: str = ""
    affixes: Tuple[str, ...] = ()
    pos_: str = ""
    morph: Dict[str, str] = field(default_factory=dict)
    is_sent_start: bool = False
    is_stop: bool = False

    def __post_init__(self) -> None:
        if not self.norm_:
            self.norm_ = normalize_word(self.text) if self.kind == "word" else self.text

    # ---- propiedades de comodidad (solo lectura) --------------------------------
    @property
    def is_punct(self) -> bool:
        return self.kind == "punct"

    @property
    def is_num(self) -> bool:
        return self.kind == "number" or self.pos_ == "NUM"

    @property
    def is_url(self) -> bool:
        return self.kind == "url"

    @property
    def is_alpha(self) -> bool:
        """``True`` si es una palabra formada solo por letras (admite guion/apóstrofo)."""
        if self.kind != "word":
            return False
        return self.norm_.replace("-", "").replace("'", "").replace("’", "").isalpha()

    @property
    def is_title(self) -> bool:
        return self.text[:1].isupper()

    @property
    def end_idx(self) -> int:
        return self.idx + len(self.text)

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Token({self.text!r})"


class Span:
    """Un trozo contiguo de un ``Doc`` (por ejemplo, una frase)."""

    def __init__(self, doc: "Doc", start: int, end: int) -> None:
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def tokens(self) -> List[Token]:
        return self.doc.tokens[self.start:self.end]

    @property
    def text(self) -> str:
        toks = self.tokens
        if not toks:
            return ""
        return self.doc.text[toks[0].idx:toks[-1].end_idx]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, i):  # type: ignore[no-untyped-def]
        return self.tokens[i]

    def __repr__(self) -> str:
        return f"Span({self.text!r})"


class Doc:
    """Resultado de procesar un texto: el texto original y su lista de ``Token``.

    Ejemplo::

        doc = esperanto_nlp.load()("La hundo kuras.")
        len(doc)           # 5 tokens: La, hundo, kuras, y el punto
        doc[1].lemma_      # "hundo"
    """

    def __init__(self, text: str, tokens: Iterable[Token]) -> None:
        self.text = text
        self.tokens: List[Token] = list(tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, i: Union[int, slice]) -> Union[Token, Span]:
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self.tokens))
            if step != 1:
                raise ValueError("Doc solo admite rebanadas con paso 1")
            return Span(self, start, max(start, stop))
        return self.tokens[i]

    def __repr__(self) -> str:
        return f"Doc({self.text[:40]!r}{'...' if len(self.text) > 40 else ''}, {len(self)} tokens)"

    @property
    def sents(self) -> Iterator[Span]:
        """Itera sobre las frases (usa la marca ``is_sent_start`` de cada token)."""
        if not self.tokens:
            return
        start = 0
        for i, tok in enumerate(self.tokens):
            if tok.is_sent_start and i > start:
                yield Span(self, start, i)
                start = i
        yield Span(self, start, len(self.tokens))

    def to_list(
        self,
        attr: str = "lemma_",
        *,
        drop_stop: bool = False,
        drop_punct: bool = False,
        drop_num: bool = False,
        keep_pos: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Lista de un atributo de cada token, con filtros opcionales.

        Es el puente hacia el aprendizaje automático: ``doc.to_list("lemma_",
        drop_stop=True, drop_punct=True)`` da las "palabras con contenido".
        """
        keep = set(keep_pos) if keep_pos is not None else None
        out: List[str] = []
        for tok in self.tokens:
            if drop_stop and tok.is_stop:
                continue
            if drop_punct and tok.is_punct:
                continue
            if drop_num and tok.is_num:
                continue
            if keep is not None and tok.pos_ not in keep:
                continue
            out.append(getattr(tok, attr))
        return out

    def to_dicts(self) -> List[Dict[str, object]]:
        """Una fila por token (útil con ``pandas.DataFrame(doc.to_dicts())``)."""
        return [
            {
                "text": t.text, "idx": t.idx, "norm": t.norm_, "lemma": t.lemma_,
                "stem": t.stem_, "root": t.root_, "affixes": t.affixes, "pos": t.pos_, "morph": dict(t.morph),
                "is_stop": t.is_stop, "is_punct": t.is_punct,
            }
            for t in self.tokens
        ]
