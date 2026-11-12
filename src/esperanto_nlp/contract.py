"""Módulo de validación y control de calidad del contrato de datos para datasets en Esperanto.

Este módulo define y verifica las especificaciones formales que debe cumplir cualquier
archivo de datos textuales antes de ingresar al pipeline de procesamiento y modelado.
Asegura la consistencia estructural del corpus, previene errores de parseo y garantiza
que los textos estén limpios de artefactos web.

Especificaciones del formato verificado:
- Codificación: UTF-8 estricto sin marca de orden de bytes (BOM).
- Estructura tabular: Formato CSV con delimitador de punto y coma (`;`).
- Columnas requeridas: Exactamente dos columnas con cabecera `text;class`.
- Granularidad: Un párrafo por fila, sin saltos de línea embebidos.
- Integridad sintáctica: El campo `text` no debe contener el delimitador (`;`).
- Limpieza textual: Sin etiquetas HTML, sin enlaces web ni citas bibliográficas (`[1]`).
- Calidad del corpus: Validación de número mínimo de clases, umbral mínimo de párrafos
  por clase, detección de desbalanceo severo y control de párrafos duplicados.

Uso básico:
    >>> from esperanto_nlp.contract import validate_file
    >>> report = validate_file("data/dataset.csv")
    >>> if report.ok:
    ...     print("Dataset válido:", report.stats)
    ... else:
    ...     print("Errores encontrados:", report.errors)
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

_HTML = re.compile(r"<[^>]+>") # Detecta etiquetas
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE) # Detecta enlaces
_REF = re.compile(r"\[\s*\d+\s*\]") # Detecta números de citas típos de Wikipedia como [1]


@dataclass
class ContractReport:
    """Informe con los resultados de la validación del contrato de datos.

    Attributes:
        errors: Lista de mensajes de error críticos que invalidan el contrato.
        warnings: Lista de advertencias sobre calidad o distribución de los datos.
        stats: Diccionario con estadísticas del corpus (número de filas, clases y media de caracteres).
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Indica si el dataset cumple estrictamente el contrato (sin errores críticos)."""
        return not self.errors

    def summary(self) -> str:
        """Genera un resumen textual formateado con errores, advertencias y estadísticas."""
        lines = [f"Contrato de datos: {'CUMPLE' if self.ok else 'NO CUMPLE'}"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  aviso: {w}" for w in self.warnings]
        lines.append(f"  estadísticas: {self.stats}")
        return "\n".join(lines)


def validate_csv_text(
    content: str, *, min_classes: int = 3, min_per_class: int = 300, max_errors: int = 25
) -> ContractReport:
    """Valida el contenido en texto de un CSV contra las reglas del contrato de datos.

    Analiza la estructura tabular, codificación sin BOM, integridad de columnas y
    limpieza del texto, calculando métricas de balanceo y volumen.

    Args:
        content: Cadena de texto completa con el contenido del archivo CSV.
        min_classes: Número mínimo requerido de clases temáticas distintas (por defecto 3).
        min_per_class: Umbral mínimo sugerido de párrafos por clase (por defecto 300).
        max_errors: Límite máximo de errores a registrar para evitar saturar la salida.

    Returns:
        ContractReport: Objeto con el estado de validación (.ok), lista de errores
        bloqueantes (.errors), advertencias no bloqueantes (.warnings) y estadísticas
        del corpus (.stats).
    """
    rep = ContractReport()

    def err(msg: str) -> None:
        if len(rep.errors) < max_errors:
            rep.errors.append(msg)
        elif len(rep.errors) == max_errors:
            rep.errors.append("... (más errores omitidos)")

    if content.startswith("\ufeff"):
        err("El archivo empieza con BOM; debe ser UTF-8 sin BOM")
        content = content.lstrip("\ufeff")
    rows = list(csv.reader(io.StringIO(content), delimiter=";"))
    if not rows:
        err("El archivo está vacío")
        return rep
    if rows[0] != ["text", "class"]:
        err(f"Cabecera incorrecta: {rows[0]} (debe ser ['text', 'class'])")
    classes: Counter = Counter()
    seen: Counter = Counter()
    total_chars = 0
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            err(f"Línea {n}: {len(row)} columnas (deben ser 2)")
            continue
        text, label = row
        if not text.strip():
            err(f"Línea {n}: texto vacío")
        if not label.strip() or label != label.strip():
            err(f"Línea {n}: clase vacía o con espacios en los extremos")
        if ";" in text:
            err(f"Línea {n}: el texto contiene ';'")
        if "\n" in text or "\r" in text:
            err(f"Línea {n}: el texto contiene saltos de línea")
        if _HTML.search(text):
            err(f"Línea {n}: contiene etiquetas HTML")
        if _URL.search(text):
            err(f"Línea {n}: contiene enlaces")
        if _REF.search(text):
            err(f"Línea {n}: contiene referencias tipo [1]")
        classes[label] += 1
        seen[text] += 1
        total_chars += len(text)
    n_rows = sum(classes.values())
    rep.stats = {
        "filas": n_rows,
        "clases": dict(sorted(classes.items())),
        "media_caracteres": round(total_chars / n_rows, 1) if n_rows else 0,
    }
    if len(classes) < min_classes:
        rep.warnings.append(f"Hay {len(classes)} clases; el enunciado pide al menos {min_classes}")
    for label, count in classes.items():
        if count < min_per_class:
            rep.warnings.append(f"Clase {label!r}: {count} párrafos (<{min_per_class} recomendados)")
    if classes and max(classes.values()) > 3 * min(classes.values()):
        rep.warnings.append("Clases desbalanceadas (la mayor tiene >3x la menor)")
    dups = sum(c - 1 for c in seen.values() if c > 1)
    if dups:
        rep.warnings.append(f"{dups} párrafos duplicados")
    return rep


def validate_file(path: Union[str, Path], **kwargs) -> ContractReport:  # type: ignore[no-untyped-def]
    """Lee un archivo desde disco y valida su cumplimiento del contrato de datos.

    Verifica en primer lugar que la codificación sea UTF-8 estricta sin errores
    de decodificación. Si la lectura es exitosa, delega la validación de estructura
    y contenido en `validate_csv_text`.

    Args:
        path: Ruta al archivo CSV en el sistema de archivos (str o Path).
        **kwargs: Parámetros opcionales transmitidos a `validate_csv_text`
            (como `min_classes`, `min_per_class`, `max_errors`).

    Returns:
        ContractReport: Objeto con los resultados, advertencias y estadísticas de la validación.
    """
    raw = Path(path).read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        rep = ContractReport()
        rep.errors.append(f"No es UTF-8 válido: {exc}")
        return rep
    return validate_csv_text(content, **kwargs)
