"""Validador del *Contrato de datos* del enunciado.

CSV UTF-8, separador ``;``, cabecera ``text;class``, un párrafo por fila, sin ``;``
dentro del texto, sin HTML, enlaces ni referencias. Lo usan el script del dataset, la
integración continua y el cuaderno (para aceptar el dataset de cualquier equipo).
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_REF = re.compile(r"\[\s*\d+\s*\]")


@dataclass
class ContractReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"Contrato de datos: {'CUMPLE' if self.ok else 'NO CUMPLE'}"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  aviso: {w}" for w in self.warnings]
        lines.append(f"  estadísticas: {self.stats}")
        return "\n".join(lines)


def validate_csv_text(
    content: str, *, min_classes: int = 3, min_per_class: int = 300, max_errors: int = 25
) -> ContractReport:
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
    raw = Path(path).read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        rep = ContractReport()
        rep.errors.append(f"No es UTF-8 válido: {exc}")
        return rep
    return validate_csv_text(content, **kwargs)
