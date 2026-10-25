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

# Windows: si la salida se redirige (> archivo.txt) Python usa cp1252 y fallaría con ĉ, ĝ, ŝ, ŭ...
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")



class ExtractError(RuntimeError):
    """La API no respondió tras varios reintentos."""


# ============================================================================
# EXTRACT: hablar con la API
# ============================================================================
def make_session(user_agent: str):  # noqa: ANN201
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
    """GET con reintentos y espera creciente (1s, 2s, 4s...). Devuelve el JSON."""
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
    """Miembros de una categoría (``page`` = artículos, ``subcat`` = subcategorías)."""
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
    """Tamaño de varias categorías (``pages`` = artículos, ``subcats`` = subcategorías).

    Usa ``prop=categoryinfo``, que acepta hasta 50 títulos por petición.
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
    """Subcategorías directas de ``category`` como ``(nombre, artículos, subcategorías)``,
    ordenadas de mayor a menor número de artículos."""
    names = [m["title"].split(":", 1)[-1] for m in category_members(session, cfg, category, "subcat")]
    info = category_info(session, cfg, names)
    rows = [(n, info.get(n, {}).get("pages", 0), info.get(n, {}).get("subcats", 0)) for n in names]
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def crawl_category(session, cfg: Dict[str, Any], root: str) -> Dict[int, str]:
    """Recorre la categoría ``root`` y sus subcategorías (BFS) hasta ``max_category_depth``.

    Devuelve ``{pageid: título}``. Se detiene al superar ``max_candidates_per_class``.
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
    """Texto plano de un artículo + id de revisión. Usa caché en disco si existe."""
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
    """Separa el texto plano en líneas-párrafo, saltando encabezados y secciones inútiles."""
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
    """Párrafos válidos de un artículo (limpios, sin duplicados, máx. N por artículo)."""
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
    """Quita los artículos presentes en más de una clase (su etiqueta sería ambigua)."""
    owners: Dict[int, int] = defaultdict(int)
    for pages in candidates.values():
        for pid in pages:
            owners[pid] += 1
    return {cls: {pid: t for pid, t in pages.items() if owners[pid] == 1}
            for cls, pages in candidates.items()}


def balance_rows(rows: List[Dict[str, Any]], classes: List[str], seed: int) -> List[Dict[str, Any]]:
    """Reduce todas las clases al tamaño de la menor (muestreo con semilla)."""
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
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wiki_api", "classes", "seed"):
        if key not in cfg:
            raise SystemExit(f"Falta la clave {key!r} en {path}")
    return cfg


def collect_candidates(session, cfg: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
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
