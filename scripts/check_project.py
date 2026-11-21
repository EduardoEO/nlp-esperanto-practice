"""Verifica la instalación, las pruebas unitarias, los datos y el cuaderno en español.

Cada paso indica qué componente del proyecto comprueba y genera un resumen final.
El registro completo de cada paso se guarda en ``salidas/<paso>.txt``.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
ANCHO = 92

# Descripción legible de cada archivo de pruebas en la tabla resumen
AREAS = OrderedDict([
    ("test_normalization_tokenizer.py", "Tokenizador y normalización del Esperanto"),
    ("test_morph_analyzer.py", "Analizador morfológico, POS y lematizador"),
    ("test_derivation.py", "Descomposición derivativa de afijos y raíces"),
    ("test_pipeline.py", "Pipeline y objetos Doc/Token/Span"),
    ("test_sklearn_wrapper.py", "Integración con scikit-learn para el modelo"),
    ("test_cleaning_contract.py", "Limpieza de párrafos y contrato de datos"),
    ("test_build_dataset.py", "Construcción del dataset desde Wikipedia"),
    ("test_noise_report.py", "Medición del ruido de etiquetas"),
    ("test_notebook_integrity.py", "Cuaderno: partición por artículo y respaldos"),
    ("test_glosario_es.py", "Glosario y explicaciones en español de las salidas"),
])


def cabecera(n, total, titulo, alcance, que):
    """Imprime un separador visual con el objetivo de cada paso de verificación.

    Args:
        n: Índice del paso actual (empezando en 1).
        total: Número total de pasos programados.
        titulo: Nombre breve del paso.
        alcance: Componente o fase del proyecto que valida.
        que: Descripción concreta de la comprobación.
    """
    print("\n" + "=" * ANCHO)
    print(f"PASO {n}/{total} · {titulo}")
    print(f"Alcance: {alcance}")
    print(f"Qué se comprueba: {que}")
    print("=" * ANCHO, flush=True)


def ejecutar(nombre, comando, env, out, mostrar=True):
    """Ejecuta un script Python; guarda su salida completa y devuelve (código, texto)."""
    lineas = []
    with (out / f"{nombre}.txt").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, *comando], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace")
        for linea in proc.stdout:
            log.write(linea)
            lineas.append(linea)
            if mostrar:
                print(linea, end="", flush=True)
        return proc.wait(), "".join(lineas)


def resumen_pruebas(texto, codigo):
    """Convierte la salida de ``pytest -v`` en una tabla por áreas, en español."""
    cuenta = OrderedDict((k, {"ok": 0, "mal": 0}) for k in AREAS)
    fallos = []
    # El nombre de una prueba puede llevar espacios o símbolos: se captura de forma no codiciosa.
    for m in re.finditer(r"^tests/([\w.]+)::(.*?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", texto, re.M):
        archivo, prueba, estado = m.groups()
        area = cuenta.setdefault(archivo, {"ok": 0, "mal": 0})
        if estado in ("PASSED", "XFAIL", "XPASS", "SKIPPED"):
            area["ok"] += 1
        else:
            area["mal"] += 1
            fallos.append(f"{archivo}::{prueba}")
    total_ok = sum(a["ok"] for a in cuenta.values())
    total_mal = sum(a["mal"] for a in cuenta.values())
    print(f"{'Área de la práctica':<62}{'superadas':>10}{'falladas':>10}")
    print("-" * ANCHO)
    for archivo, a in cuenta.items():
        if a["ok"] + a["mal"]:
            print(f"{AREAS.get(archivo, archivo):<62}{a['ok']:>10}{a['mal']:>10}")
    print("-" * ANCHO)
    print(f"{'TOTAL':<62}{total_ok:>10}{total_mal:>10}")
    # Comprobación cruzada con el recuento que da el propio pytest en su última línea
    declarado = sum(int(x) for x in re.findall(r"(\d+) (?:passed|failed|error)", texto.splitlines()[-1])) if texto.strip() else 0
    if declarado and declarado != total_ok + total_mal:
        print(f"\n  ! Aviso: pytest declara {declarado} pruebas y esta tabla recoge {total_ok + total_mal}. "
              "Revisa salidas/tests.txt.")
    if codigo and not fallos:
        print("\nPytest terminó con error pero no se reconocieron pruebas falladas: revisa salidas/tests.txt.")
    for f in fallos:
        print(f"\n  FALLA: {f}")
    if fallos:
        motivos = [l.strip() for l in texto.splitlines() if l.startswith("E ")][:6]
        print("\n  Motivo (primeras líneas del error):")
        for m in motivos:
            print("    " + m)
        print("  Detalle completo: salidas/tests.txt")
    return total_ok, total_mal


def resumen_cuaderno(out):
    """Lee ``resumen_ejecucion.json`` e imprime las métricas del cuaderno.

    Args:
        out: Directorio donde ``run_notebook.py`` guarda los artefactos de salida.
    """
    ruta = out / "resumen_ejecucion.json"
    if not ruta.exists():
        return
    r = json.loads(ruta.read_text(encoding="utf-8"))
    print("\nRESULTADO DEL CUADERNO (modelo de clasificación):")
    print(f"  · Datos: {r.get('dataset_source')} · partición por artículo: "
          f"{'sí' if r.get('grouped_by_article') else 'NO (por párrafo)'} · "
          f"entrenamiento {r.get('train_size')} / test {r.get('test_size')} párrafos")
    for m in r.get("test_metrics", []):
        print(f"  · {m['modelo']:<42} accuracy {100 * m['accuracy']:5.1f} %   F1 macro {m['f1_macro']:.3f}")
    print("    (accuracy = % de párrafos bien clasificados; F1 macro = acierto medio de las 3 clases)")
    web = r.get("web_sources", {})
    print(f"  · Páginas web clasificadas: {', '.join(f'{k} [{v}]' for k, v in web.items()) or 'ninguna'}")
    print(f"  · Tiempo total: {r.get('elapsed_seconds')} s")
    if r.get("emergency_dataset"):
        print("  ! MODO EMERGENCIA: estos números NO sirven para evaluar el modelo.")


def main():
    """Ejecuta la batería de comprobaciones y devuelve 0 si todas superan la validación.

    Returns:
        int: Código de salida ``0`` si todo pasa, o el código de error del primer paso fallido.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--web', action='store_true', help='intentar además páginas web reales')
    ap.add_argument('--skip-notebook', action='store_true', help='solo comprobaciones rápidas')
    args = ap.parse_args()
    out = ROOT / 'salidas'
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'

    pasos = [
        ('dependencias', ['-m', 'pip', 'check'],
         'Entorno de trabajo', 'las librerías instaladas son compatibles entre sí'),
        ('tests', ['-m', 'pytest', '-o', 'addopts=', '-v', '--tb=short', '-p', 'no:cacheprovider'],
         'Librería NLP · dataset ETL · cuaderno ML',
         'pruebas automáticas de la librería, del script del dataset y del cuaderno'),
        ('contrato_real', ['scripts/validate_dataset.py', 'data/dataset.csv'],
         'Contrato de datos', 'el CSV es UTF-8, tiene cabecera text;class, no contiene ";" ni HTML ni enlaces'),
        ('ruido_auto', ['scripts/noise_report.py'],
         'Ruido por heredar la etiqueta del artículo',
         'un indicador automático y exploratorio de párrafos poco característicos de su clase'),
    ]
    if not args.skip_notebook:
        pasos.append(('cuaderno', ['scripts/run_notebook.py'] + ([] if args.web else ['--offline']),
                      'Modelo ML y clasificación de páginas web',
                      'ejecuta todo el cuaderno: entrena, evalúa con test separado y clasifica páginas'))
    status = {'python': sys.version, 'executable': sys.executable,
              'started_at': datetime.now().astimezone().isoformat(), 'steps': {}}
    print(f'Python {platform.python_version()} · {sys.executable}')
    resultados = []
    for n, (nombre, comando, requisito, que) in enumerate(pasos, 1):
        cabecera(n, len(pasos), nombre.replace('_', ' ').capitalize(), requisito, que)
        if nombre == 'tests':
            codigo, texto = ejecutar(nombre, comando, env, out, mostrar=False)
            ok, mal = resumen_pruebas(texto, codigo)
            detalle = f'{ok} pruebas superadas, {mal} falladas'
        else:
            # El cuaderno vuelca un JSON largo: se guarda en salidas/ y aquí solo se muestra el resumen.
            codigo, texto = ejecutar(nombre, comando, env, out, mostrar=(nombre != 'cuaderno'))
            detalle = 'correcto' if not codigo else f'ERROR (código {codigo})'
            if nombre == 'ruido_auto':
                print('\nAviso para leer la tabla: es un indicador de VOCABULARIO, no el porcentaje de etiquetas '
                      'erróneas.\nUn párrafo "sin palabras clave" puede ser perfectamente informativo.')
            if nombre == 'cuaderno' and not codigo:
                resumen_cuaderno(out)
        status['steps'][nombre] = codigo
        (out / 'verificacion.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        resultados.append((nombre, requisito, 'CORRECTO' if not codigo else 'FALLA', detalle))
        if codigo:
            print(f'\nERROR en {nombre}. Registro completo: salidas/{nombre}.txt (código {codigo}).')
            break
    print("\n" + "=" * ANCHO)
    print("RESUMEN DE VERIFICACIÓN")
    print("=" * ANCHO)
    for nombre, requisito, estado, detalle in resultados:
        print(f"[{estado:^8}] {nombre:<14} {requisito}\n             -> {detalle}")
    if any(r[2] == 'FALLA' for r in resultados):
        return next(c for c in status['steps'].values() if c)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding='utf-8', check=True)
    (out / 'dependencias_instaladas.txt').write_text(freeze.stdout, encoding='utf-8')
    print('\nVERIFICACIÓN COMPLETADA.')
    print('Registros en la carpeta salidas/ (tests.txt, verificacion.json, resumen_ejecucion.json, ruido_auto.txt).')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
