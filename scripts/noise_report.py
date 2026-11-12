#!/usr/bin/env python3
"""Módulo de auditoría y análisis de ruido de etiquetas para datasets en Esperanto.

En la recopilación automatizada de textos a partir de enciclopedias web como Wikipedia,
los párrafos individuales heredan automáticamente la categoría asignada al artículo completo.
Este mecanismo introduce ruido de etiquetas (*label noise*): determinados párrafos contienen
información genérica, histórica o administrativa (ej. "Ĝi estis fondita en 1950...") que no
refleja de forma discriminativa el vocabulario característico de su temática.

Este módulo implementa dos enfoques complementarios para evaluar la calidad del corpus:
1. **Estimación heurística automática:** Modela las palabras más representativas de cada clase
   mediante un log-ratio suavizado de frecuencias documentales frente a las demás clases.
   Identifica como "probablemente genéricos" aquellos párrafos que carecen de cualquiera de
   las K palabras clave más distintivas de su temática.
2. **Herramienta interactiva de auditoría humana:** Permite extraer muestras estratificadas
   deterministas (`--sample N`), etiquetarlas mediante consola interactiva (`--annotate`)
   con persistencia atómica tolerante a interrupciones, y calcular métricas de ruido
   con precisión muestral (`--analyze`).

Uso:
    python scripts/noise_report.py                                # Análisis automático del dataset
    python scripts/noise_report.py --sample 50                    # Genera muestra de 50 filas por clase
    python scripts/noise_report.py --annotate                     # Interfaz de etiquetado manual
    python scripts/noise_report.py --annotate --class fiziko      # Etiquetar solo una clase
    python scripts/noise_report.py --analyze                      # Informe de la muestra etiquetada
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))   # Permite importar esperanto_nlp sin necesidad de instalación previa con pip

import esperanto_nlp  # noqa: E402


def read_dataset(path: Path) -> List[Dict[str, str]]:
    """Lee el archivo CSV del dataset y devuelve una lista de registros.

    Args:
        path: Ruta al archivo `dataset.csv` delimitado por `;`.

    Returns:
        List[Dict[str, str]]: Lista de diccionarios con las columnas `text` y `class`.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def content_lemmas(nlp, text: str) -> Set[str]:
    """Extrae el conjunto de lemas de contenido significativos de un texto.

    Procesa el texto mediante el pipeline de NLP para extraer lemas en minúsculas,
    descartando palabras vacías (*stop-words*), signos de puntuación, números
    y fragmentos residuales de longitud menor o igual a 2 caracteres.

    Args:
        nlp: Instancia cargada del pipeline lingüístico (`esperanto_nlp.load()`).
        text: Párrafo de texto en Esperanto.

    Returns:
        Set[str]: Conjunto de lemas informativos únicos presentes en el párrafo.
    """
    doc = nlp(text)
    return {l.lower() for l in doc.to_list("lemma_", drop_stop=True, drop_punct=True, drop_num=True)
            if len(l) > 2}


def keywords_per_class(docs: Dict[str, List[Set[str]]], k: int, min_df: int = 5) -> Dict[str, Set[str]]:
    """Identifica los K lemas más distintivos de cada clase mediante log-ratio suavizado.

    Calcula la frecuencia documental de cada lema dentro de la clase objetivo frente a su
    frecuencia agregada en todas las clases restantes. Aplica suavizado de Laplace (Lidstone)
    para evitar divisiones por cero y filtra términos con frecuencia documental inferior a `min_df`.

    Fórmula:
        score(w, c) = ln( P(w | c) / P(w | otras) )
        donde P(w | c) = (df_c + 0.5) / (N_c + 1.0)

    Args:
        docs: Diccionario `{nombre_clase: [conjuntos_de_lemas_por_parrafo]}`.
        k: Número máximo de palabras clave a seleccionar por cada clase.
        min_df: Frecuencia documental mínima requerida dentro de la clase (por defecto 5).

    Returns:
        Dict[str, Set[str]]: Diccionario `{nombre_clase: {lemas_top_k}}`.
    """
    df: Dict[str, Counter] = {c: Counter() for c in docs}
    for c, sets in docs.items():
        for s in sets:
            df[c].update(s)
    out: Dict[str, Set[str]] = {}
    for c in docs:
        n_c = len(docs[c])
        n_o = sum(len(v) for cc, v in docs.items() if cc != c)
        other: Counter = Counter()
        for cc in docs:
            if cc != c:
                other.update(df[cc])
        scored = []
        for lemma, f_c in df[c].items():
            if f_c < min_df:
                continue
            p_c = (f_c + 0.5) / (n_c + 1.0)
            p_o = (other[lemma] + 0.5) / (n_o + 1.0)
            scored.append((math.log(p_c / p_o), lemma))
        scored.sort(reverse=True)
        out[c] = {lemma for _, lemma in scored[:k]}
    return out


def read_sample(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Carga y valida el archivo CSV de muestra para auditoría manual de ruido.

    Verifica que el archivo contenga las cabeceras requeridas (`id`, `class`, `text`, `generic`),
    que no existan identificadores duplicados y que las etiquetas existentes sean válidas
    ('', '0', '1', '2').

    Args:
        path: Ruta al archivo `noise_sample.csv`.

    Returns:
        Tuple[List[str], List[Dict[str, str]]]: Tupla con los nombres de campo y las filas parseadas.

    Raises:
        ValueError: Si el archivo está vacío, corrupto o contiene etiquetas desconocidas.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames or []
        if not {"id", "class", "text", "generic"}.issubset(fields):
            raise ValueError("La muestra necesita id;class;text;generic")
        rows = list(reader)
    if not rows:
        raise ValueError("La muestra está vacía")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("La muestra contiene IDs duplicados")
    for r in rows:
        if None in r or any(v is None for v in r.values()):
            raise ValueError("Fila CSV incompleta o con columnas adicionales")
        r["generic"] = r["generic"].strip()
        if r["generic"] not in {"", "0", "1", "2"}:
            raise ValueError(f"Etiqueta inválida en id={r['id']}: {r['generic']!r}")
    return fields, rows


def save_sample(path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    """Guarda la muestra de anotaciones en disco utilizando sustitución atómica.

    Escribe primero en un archivo temporal en el mismo directorio y lo renombra
    atendiendo a `os.replace`. Esto previene la corrupción del CSV si el usuario
    interrumpe el proceso abruptamente (Ctrl+C o cierre de terminal) durante el guardado.

    Args:
        path: Ruta de destino final del archivo CSV.
        fields: Lista de encabezados de columna.
        rows: Lista de registros con las anotaciones actualizadas.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=";", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def annotate(path: Path, label: Optional[str] = None) -> None:
    """Inicia una sesión interactiva por consola para etiquetar párrafos de la muestra.

    Permite clasificar cada párrafo en:
        - 0: Informativo y característico de su temática asignada.
        - 1: Genérico / Ruido (ej. fechas de fundación, datos burocráticos o citas generales).
        - 2: Perteneciente a otra temática distinta a la etiqueta asignada.
        - s: Saltar el párrafo actual y conservarlo como pendiente.
        - q: Guardar progreso y salir de la herramienta.

    Persiste cada respuesta en disco de forma inmediata tras ser introducida.

    Args:
        path: Ruta al archivo `noise_sample.csv`.
        label: Nombre opcional de una clase específica para filtrar la sesión de anotación.

    Raises:
        ValueError: Si la clase especificada no existe en la muestra.
    """
    fields, rows = read_sample(path)
    if label is not None and label not in {r["class"] for r in rows}:
        raise ValueError(f"Clase desconocida: {label}")
    print("0 = informativo de su clase; 1 = genérico; 2 = otra temática; s = saltar; q = salir")
    print("Usad un solo proceso por CSV; para trabajo paralelo, usad copias separadas.")
    for row in rows:
        if row["generic"] or (label is not None and row["class"] != label):
            continue
        print(f"\nID {row['id']} | clase: {row['class']}\n{row['text']}")
        while True:
            try:
                answer = input("Etiqueta [0/1/2/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nFin. Las respuestas anteriores están guardadas.")
                return
            if answer == "q":
                return
            if answer == "s":
                break
            if answer in {"0", "1", "2"}:
                row["generic"] = answer
                save_sample(path, fields, rows)
                break
            print("Respuesta inválida. Usa 0, 1, 2, s o q.")
    print("Fin del recorrido. Los saltados siguen pendientes.")


def analyze_sample(path: Path) -> None:
    """Calcula y muestra las estadísticas de ruido a partir de las anotaciones humanas.

    Genera un desglose detallado por clase y para el total de la muestra, calculando
    las proporciones de párrafos informativos, genéricos y de temática errónea sobre el
    conjunto de muestras efectivamente revisadas.

    Args:
        path: Ruta al archivo `noise_sample.csv` con anotaciones.
    """
    _, rows = read_sample(path)
    print("Anotación manual: porcentajes sobre ANOTADOS, no sobre pendientes.")
    print("0 = informativo; 1 = genérico; 2 = otra temática; ruido = 1 + 2.")
    for label in sorted({r["class"] for r in rows}) + [None]:
        selected = [r for r in rows if label is None or r["class"] == label]
        counts = Counter(r["generic"] for r in selected)
        n = sum(counts[v] for v in ("0", "1", "2"))
        prefix = label if label is not None else "TOTAL MUESTRA"
        print(f"{prefix}: anotados={n}/{len(selected)}, pendientes={counts['']}, "
              f"informativos={counts['0']}, genéricos={counts['1']}, otra temática={counts['2']}")
        if n:
            print(f"  genéricos={100*counts['1']/n:.1f}%; otra temática={100*counts['2']/n:.1f}%; "
                  f"ruido={100*(counts['1']+counts['2'])/n:.1f}%")
        else:
            print("  Sin anotaciones: no se puede estimar el ruido manual.")
    if any(not r["generic"] for r in rows):
        print("INCOMPLETO: los porcentajes son provisionales; no extrapolar al corpus.")
    print("El total describe la muestra. Si las clases del corpus no pesan igual, ponderar por su tamaño.")
    print("La revisión humana también tiene incertidumbre; no constituye la verdad exacta del corpus.")


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada principal para el análisis de ruido y gestión de muestras.

    Gestiona los argumentos de línea de comandos para ejecutar la estimación automática
    sobre el dataset, generar nuevas muestras estratificadas o invocar los modos
    interactivos de anotación y análisis.

    Args:
        argv: Lista opcional de argumentos CLI (por defecto `sys.argv[1:]`).

    Returns:
        int: Código de salida 0 si la ejecución fue exitosa.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=_ROOT / "data" / "dataset.csv",
                    help="Ruta al archivo CSV del dataset a auditar (por defecto data/dataset.csv)")
    ap.add_argument("--top-k", type=int, default=30,
                    help="Número de palabras clave discriminantes a extraer por clase (por defecto 30)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Número de párrafos por clase a muestrear para revisión manual (por defecto 0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla determinista para el generador pseudoaleatorio (por defecto 42)")
    ap.add_argument("--out", type=Path, default=_ROOT / "data" / "noise_sample.csv",
                    help="Ruta del archivo CSV para la muestra manual (por defecto data/noise_sample.csv)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--annotate", action="store_true",
                      help="Inicia la interfaz de anotación manual en consola interactiva")
    mode.add_argument("--analyze", action="store_true",
                      help="Calcula métricas y resume los resultados de las muestras anotadas")
    ap.add_argument("--class", dest="label",
                    help="Filtra la sesión de anotación para trabajar solo sobre la clase indicada")
    args = ap.parse_args(argv)
    if args.label and not args.annotate:
        ap.error("--class requiere --annotate")
    if args.sample < 0 or args.top_k < 1:
        ap.error("--sample debe ser >= 0 y --top-k >= 1")
    if args.sample and (args.annotate or args.analyze):
        ap.error("--sample no se combina con --annotate/--analyze")
    if args.annotate or args.analyze:
        try:
            if args.annotate:
                annotate(args.out, args.label)
            else:
                analyze_sample(args.out)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        return 0
    if args.sample and args.out.exists():
        ap.error("La muestra ya existe: se conserva. Usa --annotate o elige otro archivo con --out.")

    rows = read_dataset(args.data)
    nlp = esperanto_nlp.load()
    docs: Dict[str, List[Set[str]]] = defaultdict(list)
    per_row: List[Set[str]] = []
    for r in rows:
        lem = content_lemmas(nlp, r["text"])
        docs[r["class"]].append(lem)
        per_row.append(lem)
    kws = keywords_per_class(docs, args.top_k)

    print("Indicador automático: ausencia de palabras clave; NO equivale a etiquetas incorrectas.")
    print(f"Ruido estimado (párrafos sin ninguna de las {args.top_k} palabras clave de su clase):")
    for c in sorted(docs):
        generic = sum(1 for r, lem in zip(rows, per_row) if r["class"] == c and not (lem & kws[c]))
        total = len(docs[c])
        print(f"  {c:<15} {generic:>5}/{total:<5} = {100 * generic / max(total, 1):5.1f}%   "
              f"palabras clave: {', '.join(sorted(kws[c])[:8])}...")

    if args.sample:
        rng = random.Random(args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";", lineterminator="\n")
            w.writerow(["id", "class", "text", "generic"])
            i = 0
            for c in sorted(docs):
                idx = [j for j, r in enumerate(rows) if r["class"] == c]
                for j in sorted(rng.sample(idx, min(args.sample, len(idx)))):
                    w.writerow([i, c, rows[j]["text"], ""])
                    i += 1
        print(f"\nMuestra para anotar a mano escrita en {args.out} (generic: 0 informativo, 1 genérico, 2 otra temática).")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
