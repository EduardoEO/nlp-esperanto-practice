"""Ejecuta el cuaderno de clasificación y guarda sus resultados para revisión.

El modo offline usa el dataset local y el texto web de respaldo sin descargas.
El modo remoto comprueba la obtención de la librería y el corpus desde GitHub.
La muestra de emergencia sirve para comprobar que el flujo arranca cuando
falta el corpus. Sus métricas corresponden únicamente a esa muestra.

Uso:
    python scripts/run_notebook.py --offline --out salidas

El directorio de salida conserva el cuaderno ejecutado, el registro UTF-8
por celda y el resumen JSON que genera el cuaderno al finalizar.
"""
from pathlib import Path
import argparse
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Ejecuta el cuaderno desde la raíz del proyecto con el Python activo.

    Configura el modo y el directorio de salida para el kernel de Jupyter.
    Limpia las salidas anteriores del cuaderno antes de ejecutar las celdas
    y conserva el cuaderno parcial y el registro si una celda falla.

    Returns:
        int: 0 si el cuaderno termina o 1 si falla la ejecución de una celda.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offline', action='store_true', help='datos reales locales y web de respaldo, sin descargas')
    ap.add_argument('--emergency', action='store_true', help='probar la muestra de emergencia (métricas no válidas)')
    ap.add_argument('--remote', action='store_true', help='probar el modo Colab: librería y dataset desde GitHub')
    ap.add_argument('--out', type=Path, default=ROOT / 'salidas')
    args = ap.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
    except ImportError:
        ap.error('Instala primero: python -m pip install -e ".[test,notebook,etl,runner]"')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(ROOT / 'notebooks/clasificacion_esperanto.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    summary_path = out / 'resumen_ejecucion.json'
    if summary_path.exists():
        summary_path.unlink()  # evita confundir métricas antiguas con una ejecución fallida
    env = os.environ.copy()
    env.update(ESPERANTO_MODE='remote' if args.remote else 'local',
               ESPERANTO_OFFLINE='1' if args.offline else '0',
               ESPERANTO_EMERGENCY='1' if args.emergency else '0',
               ESPERANTO_OUTPUT_DIR=str(out), PYTHONUTF8='1')
    # Kernel exactamente del entorno activo, sin registrar kernels globales.
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    def progress(cell, cell_index, **kwargs):
        """Muestra la posición de cada celda de código al iniciar su ejecución."""
        if cell.cell_type == 'code':
            print(f'Ejecutando celda {cell_index + 1}/{len(nb.cells)}...', flush=True)
    client = NotebookClient(nb, km=km, timeout=600, allow_errors=False,
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_start=progress)
    error = None
    try:
        client.execute(env=env)
    except Exception:
        error = traceback.format_exc()
    finally:
        # Con un KernelManager externo, el cliente no es su propietario.
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        nbformat.write(nb, out / 'clasificacion_esperanto_ejecutado.ipynb')
    pieces = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        pieces.append(f'\n--- CELDA {i + 1} ---\n')
        for item in cell.get('outputs', []):
            if item.output_type == 'stream':
                pieces.append(item.text)
            elif item.output_type in ('execute_result', 'display_data'):
                if 'text/plain' in item.data:
                    pieces.append(item.data['text/plain'] + '\n')
            elif item.output_type == 'error':
                pieces.append('\n'.join(item.traceback) + '\n')
    if error:
        pieces.append(error)
    text = ''.join(pieces)
    (out / 'ejecucion_cuaderno.txt').write_text(text, encoding='utf-8')
    print(text)
    print('Resultados guardados en:', out)
    return 1 if error else 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
