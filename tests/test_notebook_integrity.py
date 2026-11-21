"""Regresiones sobre funciones reales del cuaderno, sin ejecutar descargas ni entrenar."""
from pathlib import Path
import ast
import base64
import csv
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / 'notebooks/clasificacion_esperanto.ipynb'
if not NOTEBOOK_PATH.is_file():
    pytest.skip('El cuaderno de clasificación aún no está disponible.', allow_module_level=True)

np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
pytest.importorskip('sklearn')

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from esperanto_nlp.contract import validate_csv_text
from esperanto_nlp.cleaning import clean_paragraph
from sync_notebook_snapshot import snapshot

NB = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))


def functions(**overrides):
    """Carga solo las funciones del cuaderno con las dependencias de prueba."""
    env = dict(pd=pd, np=np, io=io, hashlib=hashlib, SEED=42, OFFLINE=False,
               UA='test', time=SimpleNamespace(sleep=lambda _: None),
               validate_csv_text=validate_csv_text, StratifiedGroupKFold=StratifiedGroupKFold,
               StratifiedKFold=StratifiedKFold, clean_paragraph=clean_paragraph)
    env.update(overrides)
    for cell in NB['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=defs, type_ignores=[]), '<notebook>', 'exec'), env)
    return env


def test_embedded_library_matches_source():
    """Verifica los módulos del ZIP embebido, sus rutas y su SHA-256."""
    source = next(''.join(c['source']) for c in NB['cells'] if 'EMBEDDED_LIBRARY_B64 =' in ''.join(c['source']))
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    payload = base64.b64decode(values['EMBEDDED_LIBRARY_B64'])
    # Se compara el CONTENIDO de los archivos, no los bytes del ZIP comprimido: Python 3.14 en Windows y
    # Python 3.12 en Linux comprimen distinto y el ZIP no es idéntico aunque el código sí lo sea.
    with zipfile.ZipFile(io.BytesIO(payload)) as embebido, zipfile.ZipFile(io.BytesIO(snapshot())) as actual:
        assert sorted(embebido.namelist()) == sorted(actual.namelist())
        distintos = [n for n in embebido.namelist() if embebido.read(n) != actual.read(n)]
        assert not distintos, f"la copia embebida está desactualizada: {distintos}. Ejecuta scripts/sync_notebook_snapshot.py"
    assert hashlib.sha256(payload).hexdigest() == values['EMBEDDED_LIBRARY_SHA256']
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert all(n.startswith('esperanto_nlp/') and '..' not in n for n in z.namelist())


def test_metadata_corresponds_to_real_dataset():
    """Verifica la correspondencia entre el corpus y sus metadatos."""
    df = pd.read_csv(ROOT / 'data/dataset.csv', sep=';', dtype=str)
    text = (ROOT / 'data/dataset_meta.csv').read_text(encoding='utf-8')
    meta = functions()['aligned_metadata'](text, df)
    assert meta.article_id.nunique() == 464


def test_metadata_rejects_reordered_texts_with_same_class():
    """Rechaza párrafos reordenados dentro de una misma clase."""
    df = pd.read_csv(ROOT / 'data/dataset.csv', sep=';', dtype=str)
    df.iloc[[0, 1]] = df.iloc[[1, 0]].to_numpy()
    text = (ROOT / 'data/dataset_meta.csv').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='no corresponden'):
        functions()['aligned_metadata'](text, df)


def test_cv_keeps_test_articles_out_of_inner_folds():
    """Verifica que ningún artículo se comparta entre particiones."""
    df = pd.read_csv(ROOT / 'data/dataset.csv', sep=';')
    meta = pd.read_csv(ROOT / 'data/dataset_meta.csv', sep=';')
    y, groups = df['class'].to_numpy(), meta.article_id.to_numpy()
    make_cv = functions()['make_cv']
    train, test = make_cv(y, groups)[0]
    for fit, val in make_cv(y[train], groups[train]):
        assert set(groups[train[fit]]).isdisjoint(groups[train[val]])
        assert set(groups[train[fit]]) | set(groups[train[val]]) <= set(groups[train])
        assert set(groups[train]).isdisjoint(groups[test])


def test_cv_refuses_insufficient_articles():
    """Rechaza corpus con pocos artículos para la validación cruzada."""
    with pytest.raises(ValueError, match='suficientes'):
        functions()['make_cv'](np.array(['a','a','b','b']), np.array([1,1,2,2]))


def test_bad_remote_csv_is_retried_and_not_used():
    """Reintenta la descarga y descarta una página HTML en lugar del CSV."""
    calls = []
    def get(*a, **k):
        calls.append(a)
        return SimpleNamespace(content=b'<html>error</html>', raise_for_status=lambda: None)
    env = functions(requests=SimpleNamespace(get=get))
    assert env['fetch_text'](['https://example.invalid'], validator=env['parse_dataset']) == (None, None)
    assert len(calls) == 2


def test_offline_does_not_request_network():
    """Verifica que la obtención de textos en modo offline omita la red."""
    def fail(*a, **k):
        pytest.fail('No se debe usar la red')
    env = functions(OFFLINE=True, requests=SimpleNamespace(get=fail))
    assert env['fetch_text'](['https://example.invalid']) == (None, None)


def test_etl_writes_text_hashes(tmp_path):
    """Verifica que el ETL guarde el SHA-256 de cada párrafo."""
    import build_dataset as bd
    rows = [dict(text='La ĉelo vivas.', **{'class':'biologio'}, pageid=1, title='Ĉelo', revid=2)]
    bd.write_outputs(rows, tmp_path, {'wiki_api':'fake', 'seed':42})
    meta = list(csv.DictReader((tmp_path / 'dataset_meta.csv').open(encoding='utf-8'), delimiter=';'))
    assert meta[0]['text_sha256'] == hashlib.sha256(rows[0]['text'].encode('utf-8')).hexdigest()
