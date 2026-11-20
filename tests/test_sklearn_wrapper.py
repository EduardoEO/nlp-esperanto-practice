"""Verifica el preprocesamiento que usa el clasificador de Esperanto.

Las pruebas unitarias fijan las salidas de cada representación y del
filtrado por categoría gramatical. Las de integración usan un CSV pequeño
para comprobar el flujo completo desde el texto hasta la predicción y su
compatibilidad con las operaciones de clonado y serialización del modelo.

Uso:
    python -m pytest tests/test_sklearn_wrapper.py

Si scikit-learn no está instalado, pytest omite este módulo.
"""
import csv
import pickle

import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from esperanto_nlp.sklearn_wrapper import EsperantoPreprocessor


def load_fixture(path):
    """Carga los párrafos y sus categorías del corpus de prueba.

    Args:
        path: Ruta al CSV UTF-8 con columnas ``text`` y ``class``.

    Returns:
        Una pareja de listas de textos y etiquetas en el orden del CSV.
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [r["text"] for r in rows], [r["class"] for r in rows]


def make_pipeline(**prep):
    """Ensambla el pipeline de integración con una semilla fija.

    Args:
        **prep: Opciones que se pasan a ``EsperantoPreprocessor``.

    Returns:
        Pipeline sin entrenar con preprocesamiento, TF-IDF y regresión logística.
    """
    return Pipeline([
        ("prep", EsperantoPreprocessor(**prep)),
        ("tfidf", TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def test_preprocessor_outputs():
    """Comprueba las representaciones del texto y el filtrado por categoría gramatical."""
    p = EsperantoPreprocessor(feature="lemma")
    assert p.fit_transform(["La hundoj kuras rapide."]) == ["hundo kuri rapide"]
    assert EsperantoPreprocessor(feature="stem").transform(["La hundoj kuras."]) == ["hund kur"]
    assert EsperantoPreprocessor(feature="text", drop_stop=False, lowercase=False).transform(
        ["La hundo."]) == ["La hundo"]
    assert EsperantoPreprocessor(feature="lemma", keep_pos=["NOUN"]).transform(
        ["La hundoj kuras."]) == ["hundo"]


@pytest.mark.parametrize(
    ("feature", "text", "expected"),
    [("norm", "La cxapelon.", "ĉapelon"), ("root", "La lernejo.", "lern")],
)
def test_normalized_and_root_outputs(feature, text, expected):
    """Comprueba el sistema-x y la raíz derivativa entre las representaciones admitidas."""
    assert EsperantoPreprocessor(feature=feature).fit_transform([text]) == [expected]


def test_invalid_feature():
    """Comprueba que fit rechaza una representación desconocida."""
    with pytest.raises(ValueError):
        EsperantoPreprocessor(feature="xxx").fit(["a"])


@pytest.mark.parametrize("feature", ["norm", "lemma", "stem"])
def test_pipeline_learns_fixture(mini_dataset_path, feature):
    """Comprueba que la representación se integra con TF-IDF y el clasificador.

    El acierto sobre el corpus de entrenamiento comprueba la integración.
    También se comprueba la etiqueta de una frase conocida y que las
    probabilidades predichas sumen uno.
    """
    X, y = load_fixture(mini_dataset_path)
    pipe = make_pipeline(feature=feature).fit(X, y)
    assert pipe.score(X, y) == 1.0
    pred = pipe.predict(["La tenisisto servis la pilkon en la finalo de la turniro."])
    assert pred[0] == "teniso"
    assert abs(pipe.predict_proba(["La golejisto kaptis la pilkon."]).sum() - 1) < 1e-9


def test_clone_and_pickle(mini_dataset_path):
    """Verifica el clonado y la conservación de predicciones tras serializar el pipeline."""
    X, y = load_fixture(mini_dataset_path)
    pipe = make_pipeline().fit(X, y)
    clone(pipe)
    restored = pickle.loads(pickle.dumps(pipe))
    assert list(restored.predict(X)) == list(pipe.predict(X))
