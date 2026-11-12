#!/usr/bin/env python3
"""Analiza frases en Esperanto con nuestra librería y lo explica EN ESPAÑOL.

Requisito del enunciado que cubre (apartado 4.2.1, librería de NLP): tokenizador, lematizador,
etiquetador gramatical (POS), stop-words y objetos Doc/Token.

Uso (Windows cmd / PowerShell, macOS, Linux):

    python scripts/probar_frase.py "Kio estas tio? Ili legis la librojn."   # una frase
    python scripts/probar_frase.py                                          # modo interactivo
    python scripts/probar_frase.py -f frases.txt                            # una frase por línea
    python scripts/probar_frase.py --tecnico "La hundoj kuris."             # tabla técnica original

No hace falta saber Esperanto: la tabla indica la categoría gramatical de cada palabra, su forma
base (lema), si es una palabra vacía y, cuando la conocemos, su significado. Debajo se explica cómo
se forma cada palabra (raíz + afijos + terminación). El glosario es orientativo, no un diccionario.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
for _stream in (sys.stdout, sys.stderr):
    if not _stream.isatty() and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import esperanto_nlp  # noqa: E402
from esperanto_nlp.glosario_es import explicar_token, glosa, pos_es  # noqa: E402

LINEA = "=" * 100


def mostrar_tecnico(nlp, texto: str) -> None:
    """Formato original (para quien quiera ver los atributos tal cual los guarda Token)."""
    doc = nlp(texto)
    print(f"\n>>> {texto}")
    print(f"{'texto':18}{'tipo':8}{'POS':7}{'lema':18}{'stem':14}{'root':12}{'afijos':18}stop")
    for t in doc:
        print(f"{t.text[:17]:18}{t.kind:8}{t.pos_:7}{t.lemma_[:17]:18}{t.stem_[:13]:14}"
              f"{t.root_[:11]:12}{str(t.affixes):18}{'sí' if t.is_stop else ''}")
    print(f"({len(doc)} tokens, {len(list(doc.sents))} frase(s))")


def mostrar(nlp, texto: str, explicar: bool = True) -> None:
    doc = nlp(texto)
    print(f"\n{LINEA}")
    print("REQUISITO 4.2.1 · Librería de NLP en Esperanto (tokenizador + POS + lematizador + stop-words)")
    print(f"FRASE ANALIZADA (texto en Esperanto): {texto}")
    print(LINEA)
    print("Qué ha hecho la librería: 1) separar la frase en palabras y signos (tokenizador); "
          "2) etiquetar su categoría gramatical (POS);\n"
          "3) buscar la forma base de cada palabra (lema); 4) marcar las palabras vacías "
          "(no aportan tema y se descartan).\n")
    print(f"{'nº':>3}  {'palabra':<14}{'categoría gramatical':<24}{'lema (forma base)':<18}{'¿vacía?':<8}significado")
    print("-" * 100)
    for i, t in enumerate(doc, 1):
        categoria = pos_es(t.pos_).split(" (")[0]
        significado = glosa(t.lemma_) if t.kind == "word" else ""
        print(f"{i:>3}  {t.text[:13]:<14}{categoria:<24}{t.lemma_[:17]:<18}"
              f"{'sí' if t.is_stop else '':<8}{significado or ''}")
    print(f"\nResumen: {len(doc)} piezas, {len(list(doc.sents))} frase(s), "
          f"{sum(t.is_stop for t in doc)} palabra(s) vacía(s).")
    if explicar:
        print("\nCÓMO SE FORMA CADA PALABRA (raíz + afijos + terminación):")
        vistas = set()
        for t in doc:
            clave = t.text.lower()
            if t.kind != "word" or clave in vistas:
                continue
            vistas.add(clave)
            print(f"  · {t.text}: {explicar_token(t)}")
    print("\nNota: el glosario es orientativo (hecho a mano). Si una palabra no muestra significado, "
          "la librería la analiza igualmente\npor sus terminaciones, pero no tenemos su traducción.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="frase(s) a analizar")
    ap.add_argument("-f", "--file", type=Path, help="archivo UTF-8 con una frase por línea")
    ap.add_argument("--tecnico", action="store_true", help="mostrar la tabla técnica original")
    ap.add_argument("--sin-explicacion", action="store_true", help="no explicar cómo se forma cada palabra")
    args = ap.parse_args()
    nlp = esperanto_nlp.load()

    def una(texto: str) -> None:
        if args.tecnico:
            mostrar_tecnico(nlp, texto)
        else:
            mostrar(nlp, texto, explicar=not args.sin_explicacion)

    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                una(line.strip())
    elif args.text:
        for text in args.text:
            una(text)
    else:
        print("Modo interactivo. Escribid una frase en Esperanto y pulsad Enter (línea vacía para salir).")
        while True:
            try:
                text = input("\nfrase> ").strip()
            except EOFError:
                break
            if not text:
                break
            una(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
