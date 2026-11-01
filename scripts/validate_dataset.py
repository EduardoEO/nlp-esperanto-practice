#!/usr/bin/env python3
"""Herramienta de línea de comandos para validar un archivo CSV contra el Contrato de Datos.

Este script CLI actúa como puerta de enlace de calidad independiente. Lee un archivo
CSV desde disco y ejecuta la suite de validación definida en `esperanto_nlp.contract`.
Se utiliza tanto en local para verificar la integridad del dataset generado como en el
flujo de Integración Continua (CI) de GitHub Actions para asegurar que ningún cambio
rompa las especificaciones formales del corpus.

Especificaciones evaluadas:
    - Formato CSV con codificación UTF-8 estricta (sin BOM).
    - Separador de columnas por punto y coma (`;`).
    - Estructura exacta de dos columnas: `text;class`.
    - Un párrafo de prosa por fila, sin saltos de línea internos ni el carácter `;`.
    - Limpieza de artefactos (sin HTML, sin URLs ni citas bibliográficas `[1]`).
    - Umbrales mínimos de clases y distribución de muestras por temática.

Uso:
    python scripts/validate_dataset.py data/dataset.csv
    python scripts/validate_dataset.py data/dataset.csv --min-per-class 500 --min-classes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))   # Permite importar esperanto_nlp sin necesidad de haber instalado el paquete con pip

from esperanto_nlp.contract import validate_file  # noqa: E402

# Compatibilidad multiplataforma y blindaje de codificación en Windows:
# Si la salida estándar se redirige a un archivo (> log.txt), reconfiguramos el flujo
# a UTF-8 para evitar errores 'UnicodeEncodeError' con caracteres del Esperanto.
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    """Punto de entrada para la validación por línea de comandos del contrato de datos.

    Configura el analizador de argumentos para recibir la ruta del archivo CSV y los
    umbrales de validación, invoca `validate_file` e imprime el informe estructurado
    por la salida estándar.

    Returns:
        int: Código de salida 0 si el archivo cumple estrictamente el contrato (report.ok es True);
        código 1 si se detecta cualquier infracción crítica o error de formato.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Ruta al archivo CSV a validar (ej. data/dataset.csv)")
    parser.add_argument("--min-per-class", type=int, default=300,
                        help="Número mínimo recomendado de párrafos por clase (por defecto 300)")
    parser.add_argument("--min-classes", type=int, default=3,
                        help="Número mínimo requerido de clases temáticas distintas (por defecto 3)")
    args = parser.parse_args()
    report = validate_file(args.path, min_per_class=args.min_per_class, min_classes=args.min_classes)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
