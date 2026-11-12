"""Ayudas en español para entender las salidas de la librería.

Contexto: el equipo y el profesor leen español, pero los textos del dataset están en
Esperanto. Este módulo traduce al español lo que **no** es texto del corpus: categorías
gramaticales, terminaciones, afijos y un glosario orientativo de palabras.

    >>> from esperanto_nlp.glosario_es import glosa, pos_es
    >>> glosa("kaj"), pos_es("NOUN")
    ('y', 'sustantivo')

Límites (importante para la defensa):

* ``GLOSARIO`` NO es un diccionario. Cubre las palabras gramaticales de la librería
  (artículo, pronombres, preposiciones, conjunciones, correlativos, numerales...) y un
  vocabulario científico básico escrito a mano. Si una palabra no está, simplemente no se
  traduce: nunca se inventa un significado.
* Es una ayuda para leer resultados, no forma parte del análisis lingüístico ni del modelo.
* Conviene contrastar las glosas con un diccionario (p. ej. el PIV o Vikipedio) antes de
  citarlas como definitivas.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# --------------------------------------------------------------------------------------
# Categorías gramaticales (etiquetas «Universal POS» que usa la librería)
# --------------------------------------------------------------------------------------
POS_ES: Dict[str, str] = {
    "NOUN": "sustantivo",
    "PROPN": "nombre propio",
    "VERB": "verbo",
    "AUX": "verbo auxiliar",
    "ADJ": "adjetivo",
    "ADV": "adverbio",
    "ADP": "preposición",
    "DET": "determinante (artículo/posesivo)",
    "PRON": "pronombre",
    "CCONJ": "conjunción coordinante",
    "SCONJ": "conjunción subordinante",
    "NUM": "numeral",
    "PART": "partícula",
    "INTJ": "interjección",
    "PUNCT": "puntuación",
    "SYM": "símbolo",
    "X": "otro / no reconocido",
}

# Rasgos morfológicos (token.morph) -> texto en español
FEATS_ES: Dict[Tuple[str, str], str] = {
    ("Number", "Sing"): "singular", ("Number", "Plur"): "plural",
    ("Case", "Nom"): "caso base", ("Case", "Acc"): "acusativo (-n)",
    ("Tense", "Pres"): "presente", ("Tense", "Past"): "pasado", ("Tense", "Fut"): "futuro",
    ("Mood", "Cnd"): "condicional", ("Mood", "Imp"): "imperativo",
    ("VerbForm", "Inf"): "infinitivo", ("VerbForm", "Fin"): "forma conjugada",
    ("VerbForm", "Part"): "participio",
    ("Voice", "Act"): "voz activa", ("Voice", "Pass"): "voz pasiva",
    ("Abbr", "Yes"): "abreviatura",
}

# --------------------------------------------------------------------------------------
# Terminaciones y afijos
# --------------------------------------------------------------------------------------
VERBOS_ES: Dict[str, str] = {
    "as": "verbo en presente", "is": "verbo en pasado", "os": "verbo en futuro",
    "us": "verbo en condicional", "u": "verbo en imperativo/subjuntivo", "i": "verbo en infinitivo",
}
CLASES_ES: Dict[str, str] = {"o": "sustantivo", "a": "adjetivo", "e": "adverbio"}
PARTICIPIOS_ES: Dict[str, str] = {
    "ant": "participio activo de presente (el que está haciendo)",
    "int": "participio activo de pasado (el que hizo)",
    "ont": "participio activo de futuro (el que hará)",
    "at": "participio pasivo de presente (lo que está siendo hecho)",
    "it": "participio pasivo de pasado (lo que fue hecho)",
    "ot": "participio pasivo de futuro (lo que será hecho)",
}
AFIJOS_ES: Dict[str, str] = {
    "mal-": "contrario; ej.: bona = bueno -> malbona = malo",
    "-ej": "lugar; ej.: lerni = aprender -> lernejo = escuela",
    "-ist": "persona que se dedica a algo; ej.: fiziko -> fizikisto = físico",
    "-in": "femenino; ej.: instruisto = maestro -> instruistino = maestra",
    "-ig": "hacer que sea / causar; ej.: pura = puro -> purigi = purificar",
    "-iĝ": "volverse / llegar a ser; ej.: ruĝa = rojo -> ruĝiĝi = ponerse rojo",
    "-ebl": "que se puede; ej.: legi = leer -> legebla = legible",
    "-ind": "digno de; ej.: admiri = admirar -> admirinda = admirable",
    "-aĵ": "cosa concreta; ej.: manĝi = comer -> manĝaĵo = comida",
    "-ec": "cualidad; ej.: bela = bello -> beleco = belleza",
    "-ism": "doctrina o sistema, como «-ismo»; ej.: kapitalismo",
    "-eg": "aumentativo; ej.: granda = grande -> grandega = enorme",
}

# --------------------------------------------------------------------------------------
# Glosario orientativo (lema -> español)
# --------------------------------------------------------------------------------------
_GRAMATICALES: Dict[str, str] = {
    # determinantes y posesivos
    "la": "el / la / los / las (artículo)", "ambaŭ": "ambos", "mia": "mi / mío", "via": "tu / su (de usted) / vuestro",
    "lia": "su (de él)", "ŝia": "su (de ella)", "ĝia": "su (de ello)", "nia": "nuestro", "ilia": "su (de ellos)",
    "sia": "su (de sí mismo; reflexivo)", "ia": "algún tipo de", "ies": "de alguien", "kia": "¿qué clase de?",
    "kies": "¿de quién? / cuyo", "nenia": "ningún tipo de", "nenies": "de nadie", "tia": "tal / de esa clase",
    "ties": "de ese / de aquel", "ĉia": "de todo tipo", "ĉies": "de todos",
    # preposiciones
    "al": "a / hacia", "anstataŭ": "en lugar de", "antaŭ": "antes de / delante de", "apud": "junto a",
    "da": "de (tras una cantidad)", "de": "de / desde", "dum": "durante", "ekde": "desde (un momento)",
    "ekster": "fuera de", "el": "de / desde dentro de", "en": "en / dentro de", "inter": "entre",
    "je": "preposición comodín, sin significado fijo", "kontraŭ": "contra", "krom": "excepto / además de",
    "kun": "con (compañía)", "laŭ": "según", "malantaŭ": "detrás de", "malgraŭ": "a pesar de",
    "minus": "menos (resta)", "per": "mediante / con (instrumento)", "plus": "más (suma)",
    "po": "a razón de", "por": "para", "post": "después de / detrás de", "preter": "más allá de / pasando junto a",
    "pri": "sobre / acerca de", "pro": "por (causa)", "sen": "sin", "sub": "bajo / debajo de",
    "super": "sobre / por encima de", "sur": "sobre (en contacto)", "tra": "a través de",
    "trans": "al otro lado de", "ĉe": "en / junto a", "ĉirkaŭ": "alrededor de / aproximadamente", "ĝis": "hasta",
    # conjunciones
    "kaj": "y", "aŭ": "o", "nek": "ni", "sed": "pero", "do": "pues / así que",
    "ke": "que (introduce una oración)", "kvankam": "aunque", "kvazaŭ": "como si", "ol": "que (en comparaciones)",
    "se": "si (condicional)", "ĉar": "porque",
    # partículas e interjecciones
    "jen": "he aquí", "ne": "no", "ĉi": "partícula: «este», «aquí»; tiu ĉi = este",
    "ĉu": "¿acaso? (pregunta de sí/no)", "adiaŭ": "adiós", "ha": "¡ah!", "ho": "¡oh!", "hura": "¡hurra!",
    "jes": "sí", "nu": "bueno / pues", "ve": "¡ay!",
    # adverbios no derivados
    "almenaŭ": "al menos", "ankaŭ": "también", "ankoraŭ": "todavía / aún", "antaŭhieraŭ": "anteayer",
    "apenaŭ": "apenas", "baldaŭ": "pronto", "des": "tanto (en «ju… des…»: cuanto… tanto…)", "eĉ": "incluso",
    "for": "lejos / fuera", "hieraŭ": "ayer", "hodiaŭ": "hoy", "ja": "en efecto", "jam": "ya",
    "ju": "cuanto (en «ju… des…»)", "malplej": "el menos", "malpli": "menos", "morgaŭ": "mañana",
    "nun": "ahora", "nur": "solo / solamente", "plej": "el más", "pli": "más", "plu": "más / en adelante",
    "preskaŭ": "casi", "tamen": "sin embargo", "tre": "muy", "tro": "demasiado", "tuj": "enseguida",
    # numerales
    "nul": "cero", "unu": "uno", "du": "dos", "tri": "tres", "kvar": "cuatro", "kvin": "cinco",
    "ses": "seis", "sep": "siete", "ok": "ocho", "naŭ": "nueve", "dek": "diez", "cent": "cien", "mil": "mil",
    # pronombres personales
    "mi": "yo", "vi": "tú / usted / vosotros", "li": "él", "ŝi": "ella", "ĝi": "ello / eso (cosa o animal)",
    "ni": "nosotros", "ili": "ellos / ellas", "oni": "se / uno (sujeto impersonal)", "si": "se / sí mismo (reflexivo)",
    # verbo ser/estar
    "esti": "ser / estar",
}

# Correlativos: prefijo (ki-, ti-, i-, ĉi-, neni-) + terminación (-u, -a, -e, -o, -al, -am, -el, -om, -es)
_CORRELATIVOS: Dict[str, str] = {
    "kiu": "¿quién? / ¿cuál? / que", "kio": "¿qué? / lo que", "kie": "¿dónde?", "kial": "¿por qué?",
    "kiam": "¿cuándo?", "kiel": "¿cómo? / como", "kiom": "¿cuánto?",
    "tiu": "ese / aquel", "tio": "eso", "tie": "allí", "tial": "por eso", "tiam": "entonces",
    "tiel": "así", "tiom": "tanto",
    "iu": "alguien / alguno", "io": "algo", "ie": "en algún lugar", "ial": "por alguna razón",
    "iam": "alguna vez", "iel": "de algún modo", "iom": "algo / un poco",
    "ĉiu": "cada / cada uno / todos", "ĉio": "todo", "ĉie": "en todas partes", "ĉial": "por todas las razones",
    "ĉiam": "siempre", "ĉiel": "de todos los modos", "ĉiom": "toda la cantidad",
    "neniu": "nadie / ninguno", "nenio": "nada", "nenie": "en ninguna parte", "nenial": "por ninguna razón",
    "neniam": "nunca", "neniel": "de ningún modo", "neniom": "nada (cantidad)",
}

# Vocabulario científico y general frecuente en el dataset (física, biología, química)
_CIENCIA: Dict[str, str] = {
    # general
    "scienco": "ciencia", "teorio": "teoría", "leĝo": "ley", "eksperimento": "experimento", "mezuri": "medir",
    "kalkuli": "calcular", "ekvacio": "ecuación", "nombro": "número", "kvanto": "cantidad", "formulo": "fórmula",
    "uzi": "usar", "havi": "tener", "povi": "poder", "fari": "hacer", "nomi": "nombrar / llamar",
    "trovi": "encontrar", "scii": "saber", "vidi": "ver", "doni": "dar", "preni": "tomar", "iri": "ir",
    "veni": "venir", "diri": "decir", "kuri": "correr", "nomo": "nombre", "jaro": "año", "tempo": "tiempo",
    "spaco": "espacio", "mondo": "mundo", "tero": "tierra", "suno": "sol", "luno": "luna", "planedo": "planeta",
    "stelo": "estrella", "lando": "país", "urbo": "ciudad", "homo": "ser humano", "libro": "libro",
    "granda": "grande", "malgranda": "pequeño", "nova": "nuevo", "bona": "bueno", "ĉefa": "principal",
    "unua": "primero", "dua": "segundo", "alia": "otro", "sama": "mismo", "multaj": "muchos", "multe": "mucho",
    "industrio": "industria", "direkto": "dirección", "stato": "estado", "sistemo": "sistema", "korpo": "cuerpo",
    # física
    "fiziko": "física", "fizika": "físico (adj.)", "fizikisto": "físico (persona)", "forto": "fuerza",
    "energio": "energía", "maso": "masa", "pezo": "peso", "movo": "movimiento", "rapido": "velocidad",
    "akcelo": "aceleración", "ondo": "onda", "lumo": "luz", "varmo": "calor", "temperaturo": "temperatura",
    "premo": "presión", "kampo": "campo", "partiklo": "partícula", "atomo": "átomo", "molekulo": "molécula",
    "elektrono": "electrón", "protono": "protón", "neŭtrono": "neutrón", "kvarko": "quark", "nukleo": "núcleo",
    "kvantuma": "cuántico", "elektra": "eléctrico", "magneta": "magnético", "gravito": "gravedad",
    "ŝargo": "carga (eléctrica)", "kristalo": "cristal", "frotforto": "fuerza de rozamiento",
    "radiado": "radiación", "elektro": "electricidad", "mekaniko": "mecánica", "optiko": "óptica",
    "sono": "sonido", "amplitudo": "amplitud", "frekvenco": "frecuencia", "simetrio": "simetría",
    # química
    "kemio": "química", "kemia": "químico (adj.)", "ĥemia": "químico (adj.; ortografía antigua)",
    "kemiisto": "químico (persona)", "elemento": "elemento", "metalo": "metal", "substanco": "sustancia",
    "reakcio": "reacción", "jono": "ión", "anjono": "anión", "katjono": "catión", "kombinaĵo": "compuesto",
    "acido": "ácido", "bazo": "base", "oksigeno": "oxígeno", "hidrogeno": "hidrógeno", "karbono": "carbono",
    "akvo": "agua", "gaso": "gas", "likvo": "líquido", "solido": "sólido", "solvaĵo": "disolución",
    "salo": "sal", "betono": "hormigón", "fero": "hierro", "oro": "oro", "arĝento": "plata", "kupro": "cobre",
    "zinko": "zinc", "plumbo": "plomo", "perioda": "periódico", "atomnumero": "número atómico",
    "organika": "orgánico", "oleo": "aceite", "alojo": "aleación", "oksido": "óxido", "fluorido": "fluoruro",
    # biología
    "biologio": "biología", "biologia": "biológico (adj.)", "biologo": "biólogo", "specio": "especie",
    "organismo": "organismo", "planto": "planta", "besto": "animal", "ĉelo": "célula", "mamulo": "mamífero",
    "geno": "gen", "populacio": "población", "vivo": "vida", "aparteni": "pertenecer", "epiteto": "epíteto",
    "birdo": "ave", "fiŝo": "pez", "insekto": "insecto", "arbo": "árbol", "arbaro": "bosque", "floro": "flor",
    "fungo": "hongo", "bakterio": "bacteria", "evoluo": "evolución", "ekologio": "ecología",
    "ekologia": "ecológico", "ekosistemo": "ecosistema", "biodiverseco": "biodiversidad", "diverseco": "diversidad",
    "sango": "sangre", "koro": "corazón", "cerbo": "cerebro", "muskolo": "músculo", "osto": "hueso",
    "haŭto": "piel", "semo": "semilla", "folio": "hoja", "radiko": "raíz (de planta)", "genetiko": "genética",
    "genetika": "genético", "familio": "familia (taxonomía)", "genro": "género (taxonomía)",
    "legi": "leer", "skribi": "escribir", "vivi": "vivir", "kreski": "crecer", "konsisti": "consistir",
    "ekzisti": "existir", "produkti": "producir", "ĉevalo": "caballo", "hundo": "perro", "kato": "gato", "vivanta": "vivo / viviente",
}

GLOSARIO: Dict[str, str] = {}
for _d in (_GRAMATICALES, _CORRELATIVOS, _CIENCIA):
    GLOSARIO.update({k.strip(): v for k, v in _d.items()})


# --------------------------------------------------------------------------------------
# Funciones públicas
# --------------------------------------------------------------------------------------
def glosa(lema: str) -> Optional[str]:
    """Significado orientativo de ``lema`` en español, o ``None`` si no está en el glosario."""
    return GLOSARIO.get((lema or "").lower())


def pos_es(pos: str) -> str:
    """Nombre en español de una etiqueta POS (``NOUN`` -> ``sustantivo``)."""
    return POS_ES.get(pos, pos)


def feats_es(feats: Optional[Mapping[str, str]]) -> str:
    """Rasgos morfológicos en español: ``{'Number': 'Plur'}`` -> ``plural``."""
    out = [FEATS_ES.get((k, v), f"{k}={v}") for k, v in (feats or {}).items()]
    return ", ".join(out)


def con_glosa(termino: str) -> str:
    """``especie`` entre paréntesis tras cada palabra conocida; las desconocidas quedan igual.

    Admite términos de varias palabras (los n-gramas del modelo): ``"specio homo"`` ->
    ``"specio (especie) homo (ser humano)"``.
    """
    partes = []
    for palabra in str(termino).split():
        g = glosa(palabra)
        if g:
            g = g.replace(" (", ", ").replace("(", "").replace(")", "")   # sin paréntesis anidados: «químico, adj.»
        partes.append(f"{palabra} ({g})" if g else palabra)
    return " ".join(partes)


def _terminacion(resto: str) -> List[Tuple[str, str]]:
    """Explica la terminación gramatical que queda tras la raíz (``oj`` -> -o sustantivo, -j plural)."""
    if not resto:
        return []
    if resto in VERBOS_ES:
        return [(f"-{resto}", VERBOS_ES[resto])]
    m = re.fullmatch(r"(?P<part>ant|int|ont|at|it|ot)?(?P<cls>[oae])?(?P<pl>j)?(?P<acc>n)?", resto)
    if not m or not any(m.groupdict().values()):
        return [(f"-{resto}", "terminación no analizada")]
    piezas: List[Tuple[str, str]] = []
    if m["part"]:
        piezas.append((f"-{m['part']}-", PARTICIPIOS_ES[m["part"]]))
    if m["cls"]:
        piezas.append((f"-{m['cls']}", CLASES_ES[m["cls"]]))
    if m["pl"]:
        piezas.append(("-j", "plural"))
    if m["acc"]:
        piezas.append(("-n", "acusativo: complemento directo o dirección"))
    return piezas


def descomponer(token: Any) -> List[Tuple[str, str]]:
    """Trocea una palabra en ``[(pieza, significado), ...]`` usando el análisis de la librería.

    Ejemplo (``fizikistoj``): ``[('fizik', 'raíz'), ('-ist', 'persona que se dedica...'),
    ('-o', 'sustantivo'), ('-j', 'plural')]``. Devuelve ``[]`` si no es una palabra.
    """
    if getattr(token, "kind", "word") != "word":
        return []
    norm, stem = token.norm_, token.stem_
    if not stem or not norm.startswith(stem):
        return []
    prefijos = [a for a in token.affixes if a.endswith("-")]
    sufijos = [a for a in token.affixes if a.startswith("-")]
    reconstruida = "".join(p.strip("-") for p in prefijos) + token.root_ + "".join(s.strip("-") for s in sufijos)
    piezas: List[Tuple[str, str]] = []
    if reconstruida == stem:
        piezas += [(p, AFIJOS_ES.get(p, "prefijo")) for p in prefijos]
        piezas.append((token.root_, "raíz"))
        piezas += [(s, AFIJOS_ES.get(s, "sufijo")) for s in sufijos]
    else:
        piezas.append((stem, "raíz"))
    return piezas + _terminacion(norm[len(stem):])


def explicar_token(token: Any) -> str:
    """Explicación en español de un ``Token``, en una línea."""
    kind = getattr(token, "kind", "word")
    if kind == "punct":
        return "signo de puntuación"
    if kind == "number":
        return "número"
    if kind == "url":
        return "dirección web o correo electrónico"
    if kind == "abbr":
        return "abreviatura"
    partes = [pos_es(token.pos_)]
    g = glosa(token.lemma_)
    if g:
        partes.append(f"significa «{g}»")
    piezas = descomponer(token)
    if len(piezas) > 1:
        partes.append("se forma: " + " + ".join(f"{p} ({m})" for p, m in piezas))
    elif not piezas:
        extra = feats_es(getattr(token, "morph", None))
        if extra:
            partes.append(extra)
    if getattr(token, "is_stop", False):
        partes.append("palabra vacía: se descarta en el modelo")
    return "; ".join(partes)


def glosas_de_parrafo(texto: str, maximo: int = 6, nlp: Any = None) -> str:
    """Pistas en español de un párrafo en Esperanto: sus palabras conocidas con su glosa.

    No es una traducción: solo lista (hasta ``maximo``) lemas del glosario que aparecen en el
    párrafo, sin repetir y sin palabras gramaticales muy frecuentes.
    """
    if nlp is None:
        from .pipeline import load
        nlp = load()
    vistos: List[str] = []
    for tok in nlp(texto):
        if tok.kind != "word" or tok.is_stop or tok.pos_ == "PROPN":
            continue
        lema = tok.lemma_.lower()
        if lema not in vistos and glosa(lema):
            vistos.append(lema)
    return ", ".join(f"{w} = {glosa(w)}" for w in vistos[:maximo])


def glosar_lista(palabras: Iterable[str]) -> str:
    """``con_glosa`` aplicado a una lista, separada por comas."""
    return ", ".join(con_glosa(p) for p in palabras)
