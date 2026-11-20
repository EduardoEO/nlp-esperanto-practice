"""Sincroniza la copia de esperanto_nlp que usa el cuaderno como respaldo.

Empaqueta los módulos Python de la librería en un ZIP y actualiza el contenido
en Base64 y su SHA-256 en la celda de arranque del cuaderno de clasificación.
El respaldo permite cargar la librería en Colab si falla la instalación
desde GitHub.

Uso:
    python scripts/sync_notebook_snapshot.py

Ejecuta el script después de modificar la librería para que el código
incluido en el cuaderno corresponda a la versión del repositorio.
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    """Empaqueta los módulos Python de esperanto_nlp.

    El orden de las rutas y las fechas del ZIP son fijos. Los módulos se
    guardan en UTF-8 y con saltos de línea LF para reducir diferencias entre
    plataformas.

    Returns:
        bytes: Contenido del ZIP con rutas relativas a src.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / 'src/esperanto_nlp').rglob('*.py')):
            info = zipfile.ZipInfo(path.relative_to(ROOT / 'src').as_posix(), date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_text(encoding="utf-8-sig").encode("utf-8"))
    return buf.getvalue()


def main():
    """Actualiza el ZIP embebido y su SHA-256 en el cuaderno de clasificación.

    Lee el cuaderno del repositorio, sustituye ambos valores en la celda de
    arranque y muestra el hash de la copia resultante.

    Raises:
        FileNotFoundError: Si falta el cuaderno de clasificación.
        json.JSONDecodeError: Si el cuaderno contiene JSON inválido.
    """
    path = ROOT / 'notebooks/clasificacion_esperanto.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    payload = snapshot()
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if 'EMBEDDED_LIBRARY_B64 =' in source:
            source = re.sub(r'EMBEDDED_LIBRARY_B64 = "[^"]*"',
                            'EMBEDDED_LIBRARY_B64 = "' + base64.b64encode(payload).decode() + '"', source)
            source = re.sub(r'EMBEDDED_LIBRARY_SHA256 = "[^"]*"',
                            'EMBEDDED_LIBRARY_SHA256 = "' + hashlib.sha256(payload).hexdigest() + '"', source)
            cell['source'] = source
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Copia de respaldo sincronizada:', hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    main()
