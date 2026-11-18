"""Derivación (Opción B): separar prefijos y sufijos de una raíz.

``decompose("malbonig")`` -> ``(("mal",), "bon", ("ig",))``

Idea: en Esperanto se pueden **construir palabras** pegando afijos a una raíz
(``lern-ej-o`` = "lugar donde se aprende"). Si quitamos esos afijos, familias de palabras
como ``lerni / lernejo / lernanto`` comparten la raíz ``lern`` y el clasificador ve un
vocabulario más pequeño (menos *sparsity*).

PERO sin un diccionario de raíces es imposible acertar siempre: no sabemos si ``-ul`` en
``molekulo`` es el sufijo "persona" (``riĉulo``) o parte de la raíz. Por eso:

* Solo se separan sufijos **seguros** en el vocabulario científico de la práctica.
  Se excluyen a propósito ``-ul -ar -ad -id -il -et -em -an -in`` porque rompen palabras
  como molekulo, polara, oksido, planeto, sistemo, vitamino... (ver docs/decisiones-diseno.md).
* La raíz resultante debe tener ≥ 3 letras y alguna vocal.
* Hay una lista de excepciones (``NO_SPLIT``) para los falsos positivos conocidos.

Es una heurística **mejorable**: ampliarla con evidencia (usando el dataset) es una de las
tareas propuestas al equipo.
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
    """Una raíz restante es aceptable si tiene ≥3 letras y alguna vocal."""
    return len(rest) >= 3 and any(c in VOWELS for c in rest)


def decompose(stem: str) -> Tuple[Tuple[str, ...], str, Tuple[str, ...]]:
    """Separa ``stem`` en ``(prefijos, raíz, sufijos)``.

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
