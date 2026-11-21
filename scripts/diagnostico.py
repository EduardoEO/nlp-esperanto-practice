#!/usr/bin/env python3
"""Ejecuta TODAS las comprobaciones y guarda un único informe: ``salidas/diagnostico.txt``.

    python scripts/diagnostico.py

Es la forma más rápida de enseñar «qué funciona y qué falla»: pegad (o adjuntad) el
informe completo. Cada bloque (A, B, C...) está explicado en la guía de pruebas.
Funciona igual en Windows, macOS y Linux (solo usa la librería estándar + el propio repo).
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

# Windows: si la salida se redirige (> archivo.txt) Python usa cp1252 y fallaría con ĉ, ĝ, ŝ, ŭ...
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "salidas"
OUT.mkdir(exist_ok=True)

SENTENCES = [
    "La fizikistoj mezuris la rapidon de la lumo.",
    "Kio estas tio? Mi ne scias kiom ĉio kostas.",
    "Ili legis la librojn de Zamenhof en Parizo.",
    "La malbonigitaj lernejoj ne ekzistas, sed molekuloj kaj planetoj jes.",
    "Kuracistoj kuracas; la sxipo de d-ro Zamenhof kostis 3,14 eŭrojn (ekz. 1.000), k.t.p.",
    "Vidu https://eo.wikipedia.org/wiki/Fiziko kaj skribu al a@b.org!",
]


def section(title: str) -> str:
    """Devuelve un encabezado visual en texto plano para separar bloques del informe."""
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n"


def run(cmd: list[str]) -> str:
    """Ejecuta un subproceso en UTF-8 desde la raíz del proyecto y captura su salida."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}   # Windows: evita cp1252
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    return f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}[código de salida: {p.returncode}]\n"


def block_a() -> str:
    """Bloque A: inspecciona la versión de Python, ``esperanto_nlp`` y dependencias."""
    import esperanto_nlp
    info = [f"Python {platform.python_version()} en {platform.system()} {platform.release()}",
            f"esperanto_nlp {esperanto_nlp.__version__} ({Path(esperanto_nlp.__file__).parent})"]
    for mod in ("pytest", "sklearn", "pandas", "matplotlib", "bs4", "requests"):
        try:
            m = __import__(mod)
            info.append(f"{mod:11} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            info.append(f"{mod:11} NO INSTALADO")
    return "\n".join(info) + "\n"


def block_b() -> str:
    """Bloque B: ejecuta la suite completa de ``pytest`` sin caché."""
    return run([sys.executable, "-m", "pytest", "-v", "--tb=short", "-p", "no:cacheprovider"])


def block_c() -> str:
    """Bloque C: analiza las frases de prueba y formatea sus atributos token a token."""
    import esperanto_nlp
    nlp = esperanto_nlp.load()
    out = io.StringIO()
    for text in SENTENCES:
        doc = nlp(text)
        out.write(f"\n>>> {text}\n")
        out.write(f"{'texto':16}{'tipo':7}{'POS':7}{'lema':16}{'stem':13}{'root':11}{'afijos':16}{'stop'}\n")
        for t in doc:
            out.write(f"{t.text[:15]:16}{t.kind:7}{t.pos_:7}{t.lemma_[:15]:16}{t.stem_[:12]:13}"
                      f"{t.root_[:10]:11}{str(t.affixes):16}{'sí' if t.is_stop else ''}\n")
    return out.getvalue()


def block_d() -> str:
    """Bloque D: mide el rendimiento de ``esperanto_nlp`` sobre 1.000 párrafos repetidos."""
    import esperanto_nlp
    nlp = esperanto_nlp.load()
    text = ("La fizikistoj mezuris la rapidon de la lumo kaj la kemiistoj analizis la molekulojn. " * 5).strip()
    texts = [text] * 1000
    t0 = time.perf_counter()
    n = sum(len(nlp(x)) for x in texts)
    dt = time.perf_counter() - t0
    return f"{len(texts)} párrafos, {n} tokens en {dt:.2f} s ({n / dt:,.0f} tokens/s)\n"


def block_e() -> str:
    """Bloque E: valida el contrato de ``data/dataset.csv`` y calcula el ruido automático."""
    ds = ROOT / "data" / "dataset.csv"
    if not ds.exists():
        return "data/dataset.csv NO EXISTE todavía (se genera con scripts/build_dataset.py build).\n"
    return (run([sys.executable, "scripts/validate_dataset.py", str(ds)])
            + run([sys.executable, "scripts/noise_report.py", "--data", str(ds)]))


def main() -> int:
    """Ejecuta todos los bloques de diagnóstico y guarda el informe en ``salidas/diagnostico.txt``."""
    parts = [("A. Entorno", block_a), ("B. Pruebas unitarias (pytest)", block_b),
             ("C. La librería sobre frases difíciles", block_c), ("D. Rendimiento", block_d),
             ("E. Dataset real (contrato y ruido)", block_e)]
    report = []
    for title, fn in parts:
        print(f"-> {title} ...", flush=True)
        try:
            body = fn()
        except Exception as exc:  # un bloque roto no debe impedir ver los demás
            body = f"FALLO DEL BLOQUE: {type(exc).__name__}: {exc}\n"
        report.append(section(title) + body)
    path = OUT / "diagnostico.txt"
    path.write_text("".join(report), encoding="utf-8")
    print(f"\nInforme guardado en {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
