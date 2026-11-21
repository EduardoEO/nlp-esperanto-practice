# NLP Esperanto (`esperanto-nlp` v1.0.0)

[![CI](https://github.com/EduardoEO/nlp-esperanto-practice/actions/workflows/ci.yml/badge.svg)](https://github.com/EduardoEO/nlp-esperanto-practice/actions/workflows/ci.yml)
![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema integral de Procesamiento de Lenguaje Natural (NLP) y clasificación temática para textos en **Esperanto**, desarrollado en Python puro sin librerías externas de NLP en su núcleo lingüístico.

El proyecto consta de tres componentes principales:
1. **Librería de NLP basada en reglas (`esperanto_nlp`):** Tokenizador con posiciones exactas, normalización Unicode NFC y sistema-x, etiquetador Universal POS, lematizador, descomposición derivativa de afijos (`root_`), stop-words gramaticales y explicabilidad en español con una API inspirada en spaCy (`Doc`, `Token`, `Span`).
2. **Construcción y auditoría del corpus:** Pipeline ETL reproducible desde Vikipedio (`subcats` → `explore` → `build`), validador estricto del contrato de datos `text;class` y herramienta de medición de ruido de etiquetas (`1.500` párrafos balanceados de `fiziko`, `biologio` y `kemio` sobre `464` artículos).
3. **Clasificación automática y web:** Transformador `EsperantoPreprocessor` compatible con `scikit-learn` (`Pipeline`, `clone`, `GridSearchCV`) y cuaderno Jupyter ejecutable tanto en local sin red como en Google Colab, con partición agrupada por artículo (`StratifiedGroupKFold`) para evitar fuga de datos.

---

## 📁 Estructura del Proyecto

```text
nlp-esperanto-practice/
├── .github/workflows/ci.yml           # CI con GitHub Actions (tests, cuaderno y docs)
├── .gitattributes                     # Normalización de saltos de línea (LF / CRLF)
├── .gitignore                         # Exclusiones de Git
├── LICENSE                            # Licencia MIT
├── mkdocs.yml                         # Configuración de MkDocs
├── pyproject.toml                     # Empaquetado PEP 621, extras y pytest
├── README.md                          # Documentación principal
├── data/                              # Corpus oficial, metadatos y respaldo web
│   ├── README.md                      # Inventario del corpus y contrato
│   ├── dataset.csv                    # 1.500 párrafos (fiziko, biologio, kemio)
│   ├── dataset_meta.csv               # article_id, revid y text_sha256 por fila
│   ├── dataset_stats.json             # Semilla, fechas y estadísticas de extracción
│   └── fallback_web_text.txt          # Textos de respaldo para inferencia offline
├── docs/                              # Documentación técnica para MkDocs
│   ├── index.md                       # Portada y mapa por fases y capas
│   ├── guia-rapida.md                 # Instalación y primeros pasos
│   ├── decisiones-diseno.md           # Decisiones arquitectónicas justificadas
│   ├── gramatica.md                   # Reglas morfológicas y afijos soportados
│   ├── dataset.md                     # Flujo ETL, contrato y auditoría de ruido
│   └── api.md                         # Referencia automática (mkdocstrings)
├── notebooks/
│   └── clasificacion_esperanto.ipynb  # Cuaderno de ML, ablación y scraping web
├── scripts/                           # Herramientas CLI de ETL, validación y control
│   ├── dataset_config.json            # Categorías, exclusiones y class_depth
│   ├── build_dataset.py               # ETL desde Vikipedio (subcats, explore, build)
│   ├── validate_dataset.py            # Validador CLI del contrato text;class
│   ├── noise_report.py                # Auditoría automática y muestreo de ruido
│   ├── probar_frase.py                # Analizador interactivo explicado en español
│   ├── sync_notebook_snapshot.py      # Sincroniza librería embebida en el cuaderno
│   ├── run_notebook.py                # Ejecución automática del cuaderno (--offline)
│   ├── check_project.py               # Verificación integral del proyecto
│   └── diagnostico.py                 # Informe completo de entorno y rendimiento
├── src/esperanto_nlp/                 # Librería lingüística e integración ML
│   ├── __init__.py                    # API pública (load, Doc, Token, Span, analyze)
│   ├── normalization.py               # Unicode NFC y conversión del sistema-x
│   ├── lexicon.py                     # Palabras cerradas, 45 correlativos y numerales
│   ├── _derivation.py                 # Descomposición derivativa de afijos (root_)
│   ├── _morph_analyzer.py             # Motor morfológico único con @lru_cache
│   ├── containers.py                  # Clases Doc, Span y Token estilo spaCy
│   ├── tokenizer.py                   # Tokenizador por prioridades y frases
│   ├── tagger.py                      # Etiquetador Universal POS
│   ├── lemmatizer.py                  # Lematizador canónico y raíces
│   ├── stopwords.py                   # Gestor de stop-words gramaticales
│   ├── glosario_es.py                 # Explicabilidad y glosas en español
│   ├── cleaning.py                    # Limpieza de párrafos compartida ETL/web
│   ├── contract.py                    # Validador estructural del contrato CSV
│   ├── sklearn_wrapper.py             # EsperantoPreprocessor para scikit-learn
│   └── resources/stopwords.json       # Lista curada opcional de stop-words
└── tests/                             # Suite de pruebas unitarias (188 tests, pytest)
    ├── conftest.py                    # Fixtures compartidas (nlp, mini_dataset_path)
    ├── fixtures/mini_dataset.csv      # Dataset reducido para pruebas rápidas
    ├── test_cleaning_contract.py      # Limpieza de párrafos y contrato CSV
    ├── test_build_dataset.py          # Pipeline ETL contra MediaWiki simulada
    ├── test_noise_report.py           # Auditoría de ruido y guardado atómico
    ├── test_normalization_tokenizer.py # NFC, sistema-x, offsets y oraciones
    ├── test_morph_analyzer.py         # Reglas morfológicas, léxico y participios
    ├── test_derivation.py             # Separación de afijos y modo root
    ├── test_pipeline.py               # Integración de EsperantoNLP y Doc/Token
    ├── test_glosario_es.py            # Cobertura del glosario y explicaciones
    ├── test_sklearn_wrapper.py        # Pipeline scikit-learn, clone y pickle
    └── test_notebook_integrity.py     # Snapshot, SHA-256, metadatos y no leakage
```

---

## 🚀 Instalación

### En local (modo desarrollo)
```bash
git clone https://github.com/EduardoEO/nlp-esperanto-practice.git
cd nlp-esperanto-practice
pip install -e ".[test,notebook,etl,runner,docs]"
```
*(Si solo necesitas el núcleo NLP sin dependencias externas basta `pip install -e .`; los extras disponibles en `pyproject.toml` son `etl`, `notebook`, `runner`, `test` y `docs`.)*

### En Google Colab
```bash
!pip install git+https://github.com/EduardoEO/nlp-esperanto-practice.git
```

---

## 💡 Uso Rápido

### En Python (`esperanto_nlp` y `scikit-learn`)
```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La fizikistoj mezuris la rapidon de la lumo.")

for token in doc:
    print(f"{token.text:12} | POS: {token.pos_:5} | Lema: {token.lemma_:10} | Raíz: {token.root_:8} | Stop: {token.is_stop}")

# Análisis morfológico de una palabra suelta
a = esperanto_nlp.analyze("malbonigoj")
# -> root='malbonig', base='bon', lemma='malbonigo', affixes=('mal-', '-ig')
```

Integración directa con `scikit-learn`:
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

### Por línea de comandos
```bash
# Analizar una frase con explicación morfológica en español
python scripts/probar_frase.py "La fizikistoj mezuris la energion."

# Validar el contrato del corpus y auditar ruido
python scripts/validate_dataset.py data/dataset.csv
python scripts/noise_report.py

# Ejecutar el cuaderno completo sin red
python scripts/run_notebook.py --offline
```

---

## 📋 Contrato de Datos

Todo dataset del proyecto es auditado por `esperanto_nlp.contract` (`scripts/validate_dataset.py`) bajo estas reglas:
- **Codificación:** UTF-8 estricto **sin BOM**.
- **Estructura:** CSV delimitado por `;` con cabecera exacta `text;class` y un párrafo por fila.
- **Limpieza:** Prohibido `;` dentro de `text`, etiquetas HTML, enlaces HTTP/HTTPS, plantillas `{{...}}` y citas `[1]`.
- **Prosa válida:** Longitud admisible, puntuación terminal y ratio alfabético $\ge 0.60$.

---

## 🧪 Pruebas y Verificación

```bash
# Suite completa de 188 pruebas unitarias
pytest

# Verificación integral (dependencias, pytest por áreas, contrato, ruido y cuaderno)
python scripts/check_project.py
```

---

## 📚 Documentación

| Guía | Contenido |
| :--- | :--- |
| [Índice y mapa del proyecto](docs/index.md) | Arquitectura completa por fases y capas |
| [Guía rápida](docs/guia-rapida.md) | Instalación, primeros pasos y uso con `scikit-learn` |
| [Decisiones de diseño](docs/decisiones-diseno.md) | 19 decisiones técnicas justificadas y limitaciones conocidas |
| [Gramática implementada](docs/gramatica.md) | Terminaciones, correlativos, casos trampa y afijos |
| [Dataset y ETL](docs/dataset.md) | Comandos `subcats`, `explore`, `build`, contrato y ruido |
| [Referencia de la API](docs/api.md) | Documentación técnica de módulos y clases |

---

## 👥 Autores y Licencia

Desarrollado en la asignatura *Minería de Textos* (CUNEF Universidad) por **Eduardo Estefanía**, **Ignacio Bravo** y **Alejandro Jesús Castro**.  
Distribuido bajo licencia [MIT](LICENSE).
