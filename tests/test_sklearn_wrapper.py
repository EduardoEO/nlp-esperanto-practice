"""Pruebas del preprocesador de Esperanto y su integración con scikit-learn."""
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
    """Lee textos y etiquetas del CSV de prueba."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [r["text"] for r in rows], [r["class"] for r in rows]


def make_pipeline(**prep):
    """Crea el pipeline de prueba con TF-IDF y regresión logística."""
    return Pipeline([
        ("prep", EsperantoPreprocessor(**prep)),
        ("tfidf", TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def test_preprocessor_outputs():
    """Verifica las salidas de lemma, stem y text y el filtro por POS."""
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
    """Verifica la conversión del sistema-x y la extracción de raíces."""
    assert EsperantoPreprocessor(feature=feature).fit_transform([text]) == [expected]


def test_invalid_feature():
    """Verifica que fit rechace una representación desconocida."""
    with pytest.raises(ValueError):
        EsperantoPreprocessor(feature="xxx").fit(["a"])


@pytest.mark.parametrize("feature", ["norm", "lemma", "stem"])
def test_pipeline_learns_fixture(mini_dataset_path, feature):
    """Verifica el acierto en entrenamiento, una predicción y la suma de probabilidades."""
    X, y = load_fixture(mini_dataset_path)
    pipe = make_pipeline(feature=feature).fit(X, y)
    assert pipe.score(X, y) == 1.0
    pred = pipe.predict(["La tenisisto servis la pilkon en la finalo de la turniro."])
    assert pred[0] == "teniso"
    assert abs(pipe.predict_proba(["La golejisto kaptis la pilkon."]).sum() - 1) < 1e-9


def test_clone_and_pickle(mini_dataset_path):
    """Verifica que el pipeline se pueda clonar y conserve sus predicciones tras usar pickle."""
    X, y = load_fixture(mini_dataset_path)
    pipe = make_pipeline().fit(X, y)
    clone(pipe)
    restored = pickle.loads(pickle.dumps(pipe))
    assert list(restored.predict(X)) == list(pipe.predict(X))
