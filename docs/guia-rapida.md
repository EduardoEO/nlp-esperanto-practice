# Guía rápida

## Instalación

```bash
pip install git+https://github.com/EduardoEO/nlp_esperanto_practica.git
# para desarrollo:
git clone https://github.com/EduardoEO/nlp_esperanto_practica.git
cd nlp_esperanto_practica
pip install -e ".[test,notebook,etl,docs]"
pytest
```

En un cuaderno de Colab: `!pip install git+https://github.com/EduardoEO/nlp_esperanto_practica.git`

## Uso

```python
import esperanto_nlp
nlp = esperanto_nlp.load()          # pipeline: tagger -> lemmatizer -> stopwords
doc = nlp("Cxu vi vidis la elefanton en Parizo?")   # sistema-x admitido

doc[0].text, doc[0].norm_           # ('Cxu', 'ĉu')  texto original y normalizado
doc[3].lemma_, doc[3].pos_          # ('elefanto', 'NOUN')
[s.text for s in doc.sents]         # frases
doc.to_list("lemma_", drop_stop=True, drop_punct=True)   # lemas con contenido
```

## Análisis de una sola palabra

```python
from esperanto_nlp import analyze
analyze("hundojn")   # raíz 'hund', NOUN, lema 'hundo', {'Number': 'Plur', 'Case': 'Acc'}
```

## Inspección interactiva por consola

```bash
python scripts/probar_frase.py "La fizikistoj mezuris la energion."
python scripts/probar_frase.py --tecnico "La hundoj kuris."
```

## Con scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from esperanto_nlp.sklearn_wrapper import EsperantoPreprocessor

model = Pipeline([
    ("prep", EsperantoPreprocessor(feature="lemma")),
    ("tfidf", TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)),
    ("clf", LogisticRegression(max_iter=1000)),
])
```
