#!/usr/bin/env python3
"""Construye el dataset de clasificación a partir de Vikipedio (Wikipedia en Esperanto).

Estructura ETL en tres capas (cada una solo depende de la anterior):

    EXTRACT   funciones que hablan con la API de MediaWiki (requests).
              Entrada: nombre de categoría / id de artículo. Salida: datos crudos.
    TRANSFORM limpieza y filtrado. Sin red, sin disco: funciones puras y testeables.
    LOAD      escritura determinista de ``dataset.csv`` (contrato text;class),
              ``dataset_meta.csv`` (artículo de origen) y ``dataset_stats.json``.

Uso:
    python scripts/build_dataset.py explore   # ¿son viables las categorías? (poco tráfico)
    python scripts/build_dataset.py build     # genera data/dataset.csv

Reproducibilidad: semilla fija, candidatos ordenados, ``lastrevid`` de cada artículo
guardado en los metadatos y caché local de respuestas en ``data/raw/``.

Por qué hay un CSV de metadatos: los párrafos de un mismo artículo se parecen mucho. Si
unos caen en train y otros en test, las métricas salen infladas (*data leakage*). Con el
``article_id`` el cuaderno puede dividir **por artículo**.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))   # funciona aunque el paquete no esté instalado con pip

from esperanto_nlp.cleaning import clean_paragraph, is_valid_paragraph
from esperanto_nlp.contract import validate_file

# Compatibilidad multiplataforma y blindaje de codificación en Windows:
# Por defecto, cuando la salida estándar (stdout o stderr) se redirige a un archivo o tubería
# (por ejemplo: python build_dataset.py explore > log.txt), el intérprete de Python en Windows
# utiliza la codificación predeterminada del sistema operativo (cp1252 / Windows-1252).
# En dicha codificación, los caracteres con diacríticos propios del Esperanto (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
# no existen en la tabla de caracteres y provocarían una excepción fatal 'UnicodeEncodeError'.
# Comprobamos no ser terminal interactivo (indica salida redirigida a disco o tubería)
# y reconfiguramos el flujo a UTF-8 estricto con reemplazo seguro de caracteres para garantizar
# una ejecución robusta y reproducible sin colapsos de consola.
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")



class ExtractError(RuntimeError):
    """La API no respondió tras varios reintentos."""


# ============================================================================
# EXTRACT: hablar con la API
# ============================================================================
def make_session(user_agent: str):  # noqa: ANN201
    """Crea y configura una sesión HTTP persistente para interactuar con Wikimedia.

    Utiliza `requests.Session` para mantener abiertas las conexiones subyacentes
    (HTTP Keep-Alive), acelerando notablemente las peticiones continuas a la API.
    Aplica de forma centralizada la cabecera User-Agent requerida por la Fundación
    Wikimedia para evitar bloqueos por cortafuegos.

    Args:
        user_agent: Cadena identificativa de la aplicación y contacto.

    Returns:
        requests.Session: Objeto de sesión configurado y listo para realizar consultas.
    """
    try:
        import requests  # import perezoso: las funciones de Transform no necesitan requests
    except ImportError:
        sys.exit("Falta la librería 'requests'. Instalad las dependencias del ETL con:\n"
                 '    pip install -e ".[etl]"      (o simplemente: pip install requests)')

    session = requests.Session()
    # Wikimedia exige un User-Agent descriptivo; sin él puede bloquear las peticiones.
    session.headers.update({"User-Agent": user_agent})
    return session


def api_get(session, api_url: str, params: Dict[str, Any], *, retries: int = 4,
            delay: float = 0.2, timeout: float = 20.0) -> Dict[str, Any]:
    """Ejecuta una petición GET contra la API de MediaWiki con reintentos y retroceso exponencial.

    Añade automáticamente los parámetros `format=json` y `formatversion=2` para recibir
    respuestas en formato JSON limpio. Implementa una pausa de cortesía entre peticiones
    y reintenta la conexión ante fallos transitorios de red o caídas de la API.

    Args:
        session: Instancia activa de `requests.Session`.
        api_url: URL base del endpoint de la API (`https://eo.wikipedia.org/w/api.php`).
        params: Parámetros específicos de la consulta de MediaWiki.
        retries: Número máximo de intentos antes de abortar (por defecto 4).
        delay: Pausa de cortesía en segundos entre peticiones (por defecto 0.2).
        timeout: Límite de tiempo en segundos para esperar la respuesta (por defecto 20.0).

    Returns:
        Dict[str, Any]: Diccionario con el JSON parseado de la respuesta de la API.

    Raises:
        ExtractError: Si la API responde con un error estructurado o si fallan todos los reintentos.
    """
    full = {"format": "json", "formatversion": "2", **params}
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            time.sleep(delay)
            resp = session.get(api_url, params=full, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Error de la API: {data['error']}")
            return data
        except Exception as exc:  # noqa: BLE001 - reintentamos ante cualquier fallo de red/JSON
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise ExtractError(f"Fallo tras {retries} intentos: {last}")


def category_members(session, cfg: Dict[str, Any], category: str, member_type: str) -> List[Dict[str, Any]]:
    """Obtiene los miembros (páginas o subcategorías) de una categoría de Wikipedia con paginación.

    Consulta el módulo `list=categorymembers` de la API de MediaWiki. Maneja el
    bucle de paginación (`continue`) para recuperar todos los elementos existentes
    sin importar si superan el límite máximo por petición (500 elementos).

    Args:
        session: Sesión HTTP activa.
        cfg: Diccionario de configuración con prefijo de categoría y endpoint.
        category: Nombre de la categoría (sin el prefijo `Kategorio:`).
        member_type: Tipo de miembros a buscar: `"page"` (artículos) o `"subcat"` (subcategorías).

    Returns:
        List[Dict[str, Any]]: Lista de diccionarios con la información de cada miembro
        (`pageid`, `title`, `ns`).
    """
    params: Dict[str, Any] = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"{cfg['category_prefix']}{category}", "cmtype": member_type, "cmlimit": "500",
    }
    if member_type == "page":
        params["cmnamespace"] = 0   # solo artículos (no "Diskuto:", "Ŝablono:"...)
    members: List[Dict[str, Any]] = []
    while True:
        data = api_get(session, cfg["wiki_api"], params, delay=cfg["request_delay_seconds"])
        members += data.get("query", {}).get("categorymembers", [])
        if "continue" not in data:
            return members
        params.update(data["continue"])   # paginación de MediaWiki


def category_info(session, cfg: Dict[str, Any], names: List[str]) -> Dict[str, Dict[str, int]]:
    """Consulta en lotes el volumen de páginas y subcategorías para una lista de categorías.

    Utiliza la propiedad `prop=categoryinfo` agrupando títulos en bloques de 50 en 50
    (máximo permitido por la API) para minimizar el número de peticiones de red.

    Args:
        session: Sesión HTTP activa.
        cfg: Diccionario de configuración general.
        names: Lista con los nombres de las categorías a consultar.

    Returns:
        Dict[str, Dict[str, int]]: Mapeo `{nombre_categoria: {"pages": int, "subcats": int}}`.
    """
    info: Dict[str, Dict[str, int]] = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        params = {"action": "query", "prop": "categoryinfo",
                  "titles": "|".join(f"{cfg['category_prefix']}{n}" for n in batch)}
        data = api_get(session, cfg["wiki_api"], params, delay=cfg["request_delay_seconds"])
        for page in data.get("query", {}).get("pages", []):
            ci = page.get("categoryinfo", {})            # ausente si la categoría está vacía
            info[page["title"].split(":", 1)[-1]] = {"pages": ci.get("pages", 0),
                                                     "subcats": ci.get("subcats", 0)}
    return info


def describe_subcategories(session, cfg: Dict[str, Any], category: str) -> List[Tuple[str, int, int]]:
    """Obtiene y ordena las subcategorías directas según su volumen de contenido.

    Herramienta fundamental para la exploración previa de datos. Permite identificar
    qué ramas del árbol tienen mayor peso para decidir listas de exclusión de ruido.

    Args:
        session: Sesión HTTP activa.
        cfg: Diccionario de configuración general.
        category: Nombre de la categoría raíz a inspeccionar.

    Returns:
        List[Tuple[str, int, int]]: Lista ordenada descendentemente por artículos de tuplas
        `(nombre_subcategoria, num_articulos, num_subcategorias)`.
    """
    names = [m["title"].split(":", 1)[-1] for m in category_members(session, cfg, category, "subcat")]
    info = category_info(session, cfg, names)
    rows = [(n, info.get(n, {}).get("pages", 0), info.get(n, {}).get("subcats", 0)) for n in names]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def crawl_category(session, cfg: Dict[str, Any], root: str) -> Dict[int, str]:
    """Rastrea el árbol de una categoría en anchura (BFS) aplicando límites y poda de ramas.

    Explora la categoría raíz y desciende por sus subcategorías hasta la profundidad
    indicada en `max_category_depth`. Aplica la lista de exclusiones (`exclude_subcategories`)
    para podar ramas indeseadas (como biografías o esbozos) y mantiene un registro de
    categorías visitadas para evitar bucles infinitos por referencias circulares.

    Args:
        session: Sesión HTTP activa.
        cfg: Diccionario de configuración (profundidad, exclusiones, límites).
        root: Nombre de la categoría raíz donde se inicia el rastreo.

    Returns:
        Dict[int, str]: Diccionario `{pageid: titulo_articulo}` con todos los artículos encontrados.
    """
    found: Dict[int, str] = {}
    excluded = {x.casefold() for x in cfg.get("exclude_subcategories", [])}
    visited: Set[str] = {root}
    queue = deque([(root, 0)])
    while queue and len(found) < cfg["max_candidates_per_class"]:
        cat, depth = queue.popleft()
        for page in category_members(session, cfg, cat, "page"):
            found[page["pageid"]] = page["title"]
        if depth < cfg["max_category_depth"]:
            for sub in sorted(category_members(session, cfg, cat, "subcat"), key=lambda m: m["title"]):
                name = sub["title"].split(":", 1)[-1]
                if name.casefold() in excluded:
                    continue          # p. ej. biografías: párrafos genéricos ("Li naskiĝis en...")
                if name not in visited:
                    visited.add(name)
                    queue.append((name, depth + 1))
    return found


def fetch_article(session, cfg: Dict[str, Any], pageid: int, cache_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Descarga el contenido crudo de un artículo de Wikipedia con caché en disco y trazabilidad.

    Recupera el texto de un artículo identificado por su `pageid` usando la API de MediaWiki.
    Aplica una política de caché en disco: si el archivo `{pageid}.json` ya existe localmente
    en `cache_dir`, lo carga inmediatamente sin realizar ninguna petición de red.

    Si no está en caché, consulta la API con los parámetros:
    - `prop=extracts|info`: Solicita el extracto del texto y los metadatos de la página.
    - `explaintext=1`: Pide a MediaWiki que devuelva texto plano legible, eliminando la
      sintaxis wiki pesada (plantillas, infoboxes, marcado de negritas e imágenes).
    - `pageids`: Identificador numérico del artículo.

    Además, extrae el `lastrevid` (identificador de revisión exacto) para garantizar la
    reproducibilidad científica y trazabilidad temporal del corpus, y persiste el resultado
    en disco en formato JSON con codificación UTF-8.

    Args:
        session: Instancia activa de `requests.Session` para reuso de conexiones HTTP.
        cfg: Diccionario de configuración con el endpoint de la API y pausas de cortesía.
        pageid: Identificador numérico único del artículo en MediaWiki.
        cache_dir: Directorio local (`data/raw/`) donde consultar o persistir la caché JSON.
            Si es None, no se utiliza caché en disco.

    Returns:
        Optional[Dict[str, Any]]: Diccionario con la estructura:
            - `pageid` (int): Identificador del artículo.
            - `title` (str): Título oficial de la página.
            - `revid` (int): Número de versión/revisión en Wikipedia.
            - `extract` (str): Texto plano crudo devuelto por la API.
            Devuelve None si el artículo no existe o está marcado como `missing`.
    """
    cache_file = cache_dir / f"{pageid}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    params = {"action": "query", "prop": "extracts|info", "explaintext": 1, "pageids": pageid}
    data = api_get(session, cfg["wiki_api"], params, delay=cfg["request_delay_seconds"])
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    article = {"pageid": pageid, "title": page.get("title", ""),
               "revid": page.get("lastrevid"), "extract": page.get("extract", "")}
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        cache_file.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
    return article


# ============================================================================
# TRANSFORM: limpieza (funciones puras)
# ============================================================================
_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1$")
#: Secciones sin prosa útil (listas de enlaces, bibliografía...). Se comparan en minúsculas.
SKIP_SECTIONS = {"referencoj", "eksteraj ligiloj", "vidu ankaŭ", "notoj", "bibliografio",
                 "literaturo", "fontoj", "kromaj legaĵoj", "galerio"}


def extract_paragraphs(extract: str) -> List[str]:
    """Separa el extracto de texto en párrafos de prosa, descartando encabezados y secciones no útiles.

    Analiza el texto línea a línea reconociendo la jerarquía de encabezados de MediaWiki
    (`== Sección ==`, `=== Subsección ===`). Implementa un mecanismo de seguimiento de nivel
    (`skip_level`): cuando encuentra una sección perteneciente a `SKIP_SECTIONS` (como
    referencias, enlaces externos, bibliografía o notas), descarta automáticamente todo su
    contenido y el de cualquier subsección anidada hasta que aparezca una sección nueva
    de nivel jerárquico igual o superior.

    Args:
        extract: Texto plano completo del artículo devuelto por la API de MediaWiki.

    Returns:
        List[str]: Lista de líneas correspondientes al cuerpo del artículo, libres de
        encabezados y secciones residuales.
    """
    paragraphs: List[str] = []
    skip_level: Optional[int] = None
    for line in extract.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            level, name = len(m.group(1)), m.group(2).casefold()
            if skip_level is not None and level > skip_level:
                continue                      # subsección de una sección descartada
            skip_level = level if name in SKIP_SECTIONS else None
            continue
        if skip_level is None:
            paragraphs.append(line)
    return paragraphs


def article_paragraphs(article: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Extrae, desinfecta, valida y limita los párrafos informativos de un artículo.

    Aplica el pipeline de transformación completo sobre el contenido de un artículo:
    1. Divide el texto en párrafos de prosa con `extract_paragraphs`.
    2. Desinfecta cada párrafo mediante `clean_paragraph` (elimina HTML, plantillas, enlaces, etc.).
    3. Descarta párrafos duplicados dentro del mismo artículo.
    4. Evalúa los criterios de calidad con `is_valid_paragraph` (longitud, palabras, ratio alfabético).
    5. Si el número de párrafos válidos supera `max_paragraphs_per_article`, realiza un
       muestreo determinista utilizando una semilla derivada del `pageid`. Esto evita que
       artículos muy extensos monopolicen el dataset y previene el sesgo por falta de diversidad.

    Args:
        article: Diccionario con la información del artículo (`pageid`, `title`, `extract`).
        cfg: Diccionario de configuración con límites volumétricos y semilla.

    Returns:
        List[str]: Lista de párrafos desinfectados y validados listos para el dataset.
    """
    seen: Set[str] = set()
    valid: List[str] = []
    for raw in extract_paragraphs(article["extract"]):
        text = clean_paragraph(raw)
        if text in seen:
            continue
        if is_valid_paragraph(text, min_chars=cfg["min_chars"], max_chars=cfg["max_chars"],
                              min_words=cfg["min_words"]):
            seen.add(text)
            valid.append(text)
    limit = cfg["max_paragraphs_per_article"]
    if len(valid) > limit:
        # Muestreo con semilla propia de cada artículo -> resultado estable entre ejecuciones.
        rng = random.Random(f"{cfg['seed']}-{article['pageid']}")
        keep = sorted(rng.sample(range(len(valid)), limit))
        valid = [valid[i] for i in keep]
    return valid


def drop_ambiguous(candidates: Dict[str, Dict[int, str]]) -> Dict[str, Dict[int, str]]:
    """Descarta artículos interdisciplinares presentes en más de una clase temática.

    En Wikipedia, algunos artículos pertenecen simultáneamente a varias categorías
    (por ejemplo, un artículo de biofísica catalogado en Biología y Física).
    Asignarle una sola clase introduciría ruido y contradicciones al clasificador.
    Esta función cuenta cuántas clases reclaman cada `pageid` y elimina por completo
    aquellos que aparezcan en dos o más clases, garantizando que el corpus contenga
    únicamente textos con asignación temática unívoca.

    Args:
        candidates: Diccionario `{nombre_clase: {pageid: titulo_articulo}}`.

    Returns:
        Dict[str, Dict[int, str]]: Mismo diccionario filtrado, conservando solo artículos
        exclusivos de una única clase.
    """
    owners: Dict[int, int] = defaultdict(int)
    for pages in candidates.values():
        for pid in pages:
            owners[pid] += 1
    return {cls: {pid: t for pid, t in pages.items() if owners[pid] == 1}
            for cls, pages in candidates.items()}


def balance_rows(rows: List[Dict[str, Any]], classes: List[str], seed: int) -> List[Dict[str, Any]]:
    """Equilibra el número de filas por clase al tamaño de la clase minoritaria.

    Garantiza que el conjunto de datos esté perfectamente balanceado entre las distintas
    disciplinas científicas, evitando sesgos hacia la temática más abundante en Wikipedia.
    Si una clase tiene más párrafos que la menor, realiza un submuestreo determinista
    ordenado mediante una semilla específica para esa clase.

    Args:
        rows: Lista de diccionarios de párrafos con su campo `class`.
        classes: Lista de nombres de las clases temáticas objetivo.
        seed: Semilla numérica para inicializar el generador pseudoaleatorio.

    Returns:
        List[Dict[str, Any]]: Lista balanceada con exactamente el mismo número de párrafos
        por clase.
    """
    by_class: Dict[str, List[Dict[str, Any]]] = {c: [r for r in rows if r["class"] == c] for c in classes}
    smallest = min(len(v) for v in by_class.values()) if by_class else 0
    out: List[Dict[str, Any]] = []
    for c in classes:
        items = by_class[c]
        if len(items) > smallest:
            rng = random.Random(f"{seed}-balance-{c}")
            items = [items[i] for i in sorted(rng.sample(range(len(items)), smallest))]
        out += items
    return out


# ============================================================================
# LOAD: escritura determinista
# ============================================================================
def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, cfg: Dict[str, Any]) -> None:
    """Persiste en disco el dataset etiquetado, los metadatos de trazabilidad y las estadísticas.

    Genera tres archivos deterministas en `out_dir`:
    1. `dataset.csv`: El archivo principal exigido por el contrato de datos (UTF-8, delimitador `;`,
       cabecera `text;class`), con un párrafo limpio por fila.
    2. `dataset_meta.csv`: Metadatos detallados para cada fila (`row`, `article_id`, `title`,
       `revid`, `class`, `text_sha256`). El identificador de artículo (`article_id`) es vital
       para permitir particiones train/test agrupadas por artículo y evitar *data leakage*.
       El hash SHA-256 asegura la integridad criptográfica de cada texto.
    3. `dataset_stats.json`: Resumen estadístico del corpus en JSON (marca temporal UTC,
       recuento de párrafos por temática, número de artículos de origen y copia de la configuración).

    Args:
        rows: Lista de diccionarios de filas validadas y balanceadas.
        out_dir: Directorio de destino en disco (típicamente `data/`).
        cfg: Diccionario de configuración utilizado en la extracción.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["text", "class"])
        for r in rows:
            w.writerow([r["text"], r["class"]])
    with open(out_dir / "dataset_meta.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["row", "article_id", "title", "revid", "class", "text_sha256"])
        for i, r in enumerate(rows):
            w.writerow([i, r["pageid"], r["title"], r["revid"], r["class"], hashlib.sha256(r["text"].encode("utf-8")).hexdigest()])
    counts: Dict[str, int] = defaultdict(int)
    articles: Dict[str, Set[int]] = defaultdict(set)
    for r in rows:
        counts[r["class"]] += 1
        articles[r["class"]].add(r["pageid"])
    stats = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": cfg["wiki_api"],
        "seed": cfg["seed"],
        "paragraphs_per_class": dict(counts),
        "articles_per_class": {c: len(a) for c, a in articles.items()},
        "config": {k: v for k, v in cfg.items() if k != "user_agent"},
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ============================================================================
# Orquestación
# ============================================================================
def load_config(path: Path) -> Dict[str, Any]:
    """Carga y valida el archivo de configuración JSON del dataset.

    Args:
        path: Ruta al archivo `dataset_config.json`.

    Returns:
        Dict[str, Any]: Configuración parseada.

    Raises:
        SystemExit: Si faltan claves obligatorias (`wiki_api`, `classes` o `seed`).
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wiki_api", "classes", "seed"):
        if key not in cfg:
            raise SystemExit(f"Falta la clave {key!r} en {path}")
    return cfg


def collect_candidates(session, cfg: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    """Recopila los artículos candidatos para cada clase y descarta los ambiguos.

    Rastrea las categorías raíz de cada clase aplicando su profundidad correspondiente
    (priorizando `class_depth` si está definida, o `max_category_depth` en su defecto).
    Aplica `drop_ambiguous` para eliminar de raíz los artículos compartidos por varias clases.

    Args:
        session: Sesión HTTP activa.
        cfg: Configuración con clases, endpoints y profundidades.

    Returns:
        Dict[str, Dict[int, str]]: Mapeo `{clase: {pageid: titulo}}` con artículos no ambiguos.
    """
    cands: Dict[str, Dict[int, str]] = {}
    for cls, roots in cfg["classes"].items():
        pages: Dict[int, str] = {}
        # Profundidad propia de la clase ("class_depth" en el JSON) o la global.
        depth = cfg.get("class_depth", {}).get(cls, cfg["max_category_depth"])
        for root in roots:
            pages.update(crawl_category(session, {**cfg, "max_category_depth": depth}, root))
        cands[cls] = pages
        print(f"  [{cls}] {len(pages)} artículos candidatos")
    return drop_ambiguous(cands)


def build_rows(session, cfg: Dict[str, Any], cache_dir: Optional[Path], limit_articles: Optional[int] = None) -> List[Dict[str, Any]]:
    """Motor orquestador del ETL: extrae, procesa y ensambla todas las filas del dataset.

    Coordina todo el flujo de generación:
    1. Identifica artículos candidatos no ambiguos con `collect_candidates`.
    2. Baraja aleatoriamente los artículos de cada clase con la semilla fija.
    3. Descarga cada artículo (con caché en `cache_dir`) y extrae párrafos válidos.
    4. Garantiza unicidad global: ningún párrafo se repite en todo el corpus (`seen_texts`).
    5. Se detiene al alcanzar `target_paragraphs_per_class` por temática.
    6. Aplica `balance_rows` para asegurar clases perfectamente equilibradas.

    Args:
        session: Sesión HTTP activa.
        cfg: Configuración con parámetros de calidad y límites.
        cache_dir: Directorio local para la caché JSON (`data/raw/`).
        limit_articles: Límite opcional de artículos por clase para pruebas rápidas.

    Returns:
        List[Dict[str, Any]]: Lista de diccionarios con los párrafos y sus metadatos.
    """
    cands = collect_candidates(session, cfg)
    rows: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()
    for cls in cfg["classes"]:
        pageids = sorted(cands[cls])
        random.Random(f"{cfg['seed']}-{cls}").shuffle(pageids)
        n_par, n_art = 0, 0
        for pid in pageids:
            if n_par >= cfg["target_paragraphs_per_class"] or (limit_articles and n_art >= limit_articles):
                break
            try:
                article = fetch_article(session, cfg, pid, cache_dir)
            except ExtractError as exc:
                print(f"    aviso: artículo {pid} omitido ({exc})", file=sys.stderr)
                continue
            if not article:
                continue
            paragraphs = [p for p in article_paragraphs(article, cfg) if p not in seen_texts]
            if not paragraphs:
                continue
            n_art += 1
            for p in paragraphs:
                seen_texts.add(p)
                rows.append({"text": p, "class": cls, "pageid": pid,
                             "title": article["title"], "revid": article["revid"]})
                n_par += 1
        print(f"  [{cls}] {n_par} párrafos de {n_art} artículos")
    if cfg.get("balance_classes", True) and rows:
        rows = balance_rows(rows, list(cfg["classes"]), cfg["seed"])
    return rows


def cmd_explore(args: argparse.Namespace) -> int:
    """Ejecuta el subcomando 'explore': comprueba la viabilidad de las temáticas en Vikipedio.

    Realiza un muestreo ligero sin descargar el corpus completo:
    1. Recopila artículos candidatos por clase y descarta los ambiguos.
    2. Imprime ejemplos aleatorios de títulos encontrados.
    3. Descarga una muestra reducida de artículos de sondeo (`args.probe`).
    4. Estima la media de párrafos válidos por artículo y proyecta si se superará
       holgadamente el mínimo de 300 párrafos recomendado por clase.

    Args:
        args: Argumentos parseados de la línea de comandos (`config`, `out`, `probe`, `depth`).

    Returns:
        int: 0 si la exploración finalizó correctamente.
    """
    cfg = _apply_overrides(load_config(args.config), args)
    session = make_session(cfg["user_agent"])
    print("Explorando categorías (comprobación de viabilidad)...")
    cands = collect_candidates(session, cfg)
    cache_dir = args.out / "raw"
    for cls, pages in cands.items():
        titles = sorted(pages.values())
        sample = random.Random(f"{cfg['seed']}-{cls}-ejemplos").sample(titles, min(8, len(titles)))
        print(f"\n[{cls}] {len(pages)} artículos tras quitar ambiguos. Ejemplos al azar: {sample}")
        if not pages:
            print("   !! 0 artículos: ¿existe 'Kategorio:...'? Revisad el nombre en eo.wikipedia.org")
            continue
        probe = sorted(pages)[: args.probe]
        counts = []
        for pid in probe:
            art = fetch_article(session, cfg, pid, cache_dir)
            counts.append(len(article_paragraphs(art, cfg)) if art else 0)
        if counts:
            avg = sum(counts) / len(counts)
            print(f"   media de párrafos válidos por artículo (muestra {len(counts)}): {avg:.1f}"
                  f" -> estimación total ≈ {avg * len(pages):.0f} (objetivo: >300)")
    return 0


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Aplica opciones de línea de comandos que pisan la configuración del JSON."""
    if getattr(args, "depth", None) is not None:
        cfg["max_category_depth"] = args.depth
        cfg["class_depth"] = {}            # --depth manda sobre la profundidad por clase
    return cfg


def cmd_subcats(args: argparse.Namespace) -> int:
    """Ejecuta el subcomando 'subcats': lista las subcategorías y su tamaño para decidir exclusiones.

    Inspecciona las subcategorías directas de las raíces configuradas y las muestra
    ordenadas por cantidad de páginas, marcando las que ya están en `exclude_subcategories`.
    Permite auditar el árbol de categorías y detectar ramas ruidosas (como biografías o esbozos).

    Args:
        args: Argumentos parseados de la línea de comandos (`config`, `categories`, `top`).

    Returns:
        int: 0 tras imprimir el listado.
    """
    cfg = load_config(args.config)
    session = make_session(cfg["user_agent"])
    excluded = {x.casefold() for x in cfg.get("exclude_subcategories", [])}
    roots = args.categories or [c for cats in cfg["classes"].values() for c in cats]
    for root in roots:
        rows = describe_subcategories(session, cfg, root)
        print(f"\n[{root}] {len(rows)} subcategorías directas (de más a menos artículos):")
        for name, pages, subs in rows[: args.top]:
            mark = "   <- ya excluida" if name.casefold() in excluded else ""
            print(f"   {pages:5d} art. {subs:4d} subkat.  {name}{mark}")
        if len(rows) > args.top:
            print(f"   ... y {len(rows) - args.top} más (usad --top para ver más)")
    print('\nPara excluir una: añadid su nombre EXACTO a "exclude_subcategories" en el JSON.')
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Ejecuta el subcomando 'build': construye el dataset completo y valida el contrato.

    Orquesta todo el proceso de recopilación, transformación, balanceo y persistencia.
    Al finalizar, ejecuta `validate_file` sobre el `dataset.csv` generado para asegurar
    que cumple estrictamente con el contrato de datos antes de finalizar.

    Args:
        args: Argumentos parseados de la línea de comandos (`config`, `out`, `depth`, `limit_articles`).

    Returns:
        int: 0 si el dataset se generó y cumplió el contrato; 1 en caso contrario.
    """
    cfg = _apply_overrides(load_config(args.config), args)
    session = make_session(cfg["user_agent"])
    print("Construyendo dataset...")
    rows = build_rows(session, cfg, args.out / "raw", args.limit_articles)
    if not rows:
        print("No se obtuvo ningún párrafo.", file=sys.stderr)
        return 1
    write_outputs(rows, args.out, cfg)
    report = validate_file(args.out / "dataset.csv")
    print(report.summary())
    return 0 if report.ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada principal para la interfaz de línea de comandos (CLI).

    Configura el analizador de argumentos `argparse` con los tres subcomandos
    disponibles (`explore`, `build`, `subcats`) y captura errores de red `ExtractError`
    para emitir diagnósticos amigables.

    Args:
        argv: Lista opcional de argumentos de línea de comandos (por defecto `sys.argv[1:]`).

    Returns:
        int: Código de salida para el sistema operativo (0 = éxito, >0 = error).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (("explore", cmd_explore, "comprobar viabilidad de las categorías"),
                            ("build", cmd_build, "generar el dataset")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", type=Path, default=_ROOT / "scripts" / "dataset_config.json")
        p.add_argument("--out", type=Path, default=_ROOT / "data")
        p.add_argument("--depth", type=int, default=None,
                       help="profundidad de subcategorías (pisa max_category_depth del JSON)")
        if name == "explore":
            p.add_argument("--probe", type=int, default=10, help="artículos a descargar por clase")
        else:
            p.add_argument("--limit-articles", type=int, default=None, help="máx. artículos por clase (pruebas)")
        p.set_defaults(func=fn)
    sc = sub.add_parser("subcats", help="listar subcategorías (y su tamaño) para decidir qué excluir")
    sc.add_argument("categories", nargs="*", help="categorías a listar (por defecto, las de la config)")
    sc.add_argument("--config", type=Path, default=_ROOT / "scripts" / "dataset_config.json")
    sc.add_argument("--top", type=int, default=40, help="máx. subcategorías a mostrar por categoría")
    sc.set_defaults(func=cmd_subcats)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ExtractError as exc:       # sin red, API caída, categoría inexistente...
        print(f"\nERROR de red/API: {exc}\n"
              "Comprobad vuestra conexión y que `wiki_api` en la config es correcta. "
              "Si Vikipedio está caída, reintentad más tarde (hay caché en data/raw/).",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
