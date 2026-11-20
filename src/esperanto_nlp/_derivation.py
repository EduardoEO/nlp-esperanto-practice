"""Descomposición derivativa de afijos en Esperanto (extracción de raíz primaria).

Separa prefijos y sufijos derivativos de una raíz flexiva (``stem``) para reducir
la dispersión léxica (*sparsity*) en tareas de clasificación y minería de textos:

``decompose("malbonig")`` -> ``(("mal",), "bon", ("ig",))``

En Esperanto se forman familias de palabras combinando afijos sobre una raíz común
(``lern-ej-o`` = "lugar donde se aprende"). Al aislar la raíz primaria, formas como
``lerni``, ``lernejo`` y ``lernanto`` convergen en la raíz ``lern``.

Sin un diccionario completo de raíces, ciertos sufijos coinciden con terminaciones
léxicas nativas (p. ej. ``-ul`` en ``molekulo`` o ``-id`` en ``oksido``). Para evitar
falsas segmentaciones sobre vocabulario científico y general:

* Solo se separan afijos de alta precisión (``SUFFIXES`` y ``PREFIXES``), excluyendo
  terminaciones ambiguas como ``-ul``, ``-ar``, ``-ad``, ``-id``, ``-il``, ``-et``,
  ``-em``, ``-an`` o ``-in`` (salvo ``-in`` tras ``-ist``, como en ``instruistino``).
* La raíz residual debe conservar al menos 3 caracteres y una vocal.
* La lista ``NO_SPLIT`` protege raíces conocidas que coinciden superficialmente con afijos.
"""
from __future__ import annotations

from typing import Tuple

from .normalization import VOWELS

#: Sufijos derivativos que se separan (de más largo a más corto).
SUFFIXES: Tuple[str, ...] = ("ebl", "ind", "ist", "ism", "ec", "ej", "eg", "ig", "iĝ", "aĵ")
#: ``-in`` (femenino) solo se separa justo después de ``-ist`` (instruistino).
PREFIXES: Tuple[str, ...] = ("mal",)

#: Raíces completas que NO deben descomponerse (parecen derivadas pero no lo son).
NO_SPLIT = frozenset({
    # -ist- que es parte de la raíz de un verbo
    "ekzist", "resist", "persist", "asist", "konsist", "insist", "desist", "subsist",
    # -eg- / -ig- / -ec- dentro de la raíz
    "koleg", "norveg", "kolegi", "intrig", "ŝanĝ", "strateg", "privileg", "kolig",
    "ĉirkaŭec", "ĝenerec",
    # mal- que no es el prefijo "contrario"
    "malari", "malt", "malajzi", "malaŭi", "maldiv",
})


def _ok(rest: str) -> bool:
    """Comprueba si la raíz residual es válida tras recortar un afijo.

    Args:
        rest: Subcadena candidata a raíz residual.

    Returns:
        bool: ``True`` si tiene al menos 3 caracteres y contiene alguna vocal.
    """
    return len(rest) >= 3 and any(c in VOWELS for c in rest)


def decompose(stem: str) -> Tuple[Tuple[str, ...], str, Tuple[str, ...]]:
    """Separa ``stem`` en una tupla ``(prefijos, raíz, sufijos)``.

    Args:
        stem: Raíz flexiva ya desprovista de terminaciones gramaticales.

    Returns:
        Tuple[Tuple[str, ...], str, Tuple[str, ...]]: Tupla con la secuencia de
            prefijos extraídos, la raíz primaria resultante y los sufijos en orden.

    >>> decompose("lernej")
    ((), 'lern', ('ej',))
    >>> decompose("malbonig")
    (('mal',), 'bon', ('ig',))
    >>> decompose("molekul")      # no se toca: -ul no está en la lista segura
    ((), 'molekul', ())
    """
    if stem in NO_SPLIT:
        return (), stem, ()
    prefixes: list = []
    suffixes: list = []
    rest = stem

    for pre in PREFIXES:
        if rest.startswith(pre) and _ok(rest[len(pre):]) and rest not in NO_SPLIT:
            prefixes.append(pre)
            rest = rest[len(pre):]
            break

    for _ in range(3):  # como mucho 3 sufijos encadenados (san-ig-ej-)
        if rest in NO_SPLIT:
            break
        cut = None
        # femenino -in tras -ist (instruist-in)
        if rest.endswith("istin") and _ok(rest[:-5]):
            cut = ("ist", 5, "in")
        else:
            for suf in SUFFIXES:
                if rest.endswith(suf) and _ok(rest[: -len(suf)]):
                    cut = (suf, len(suf), None)
                    break
        if cut is None:
            break
        suf, size, extra = cut
        if extra:
            suffixes[:0] = [suf, extra]
        else:
            suffixes.insert(0, suf)
        rest = rest[:-size]
    return tuple(prefixes), rest, tuple(suffixes)
