#!/usr/bin/env python3
"""Valida un CSV contra el contrato de datos. Uso: python scripts/validate_dataset.py data/dataset.csv"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))   # funciona aunque el paquete no esté instalado con pip

from esperanto_nlp.contract import validate_file  # noqa: E402

# Windows: si la salida se redirige (> archivo.txt) Python usa cp1252 y fallaría con ĉ, ĝ, ŝ, ŭ...
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--min-per-class", type=int, default=300)
    parser.add_argument("--min-classes", type=int, default=3)
    args = parser.parse_args()
    report = validate_file(args.path, min_per_class=args.min_per_class, min_classes=args.min_classes)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
