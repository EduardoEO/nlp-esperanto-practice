# esperanto-nlp

Librería de **NLP para Esperanto basada en reglas**, con una API inspirada en spaCy (`Doc`, `Token`, `Span`) y construida desde cero en Python puro, sin dependencias de librerías externas de NLP. Aprovecha la regularidad morfológica del Esperanto para resolver tokenización, etiquetado gramatical (Universal POS), lematización, descomposición derivativa de afijos y detección de palabras vacías.

```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La hundoj kuras rapide.")
for t in doc:
    print(t.text, t.pos_, t.lemma_, t.root_, t.is_stop)
```

---

## Mapa del proyecto por fases y capas

El proyecto está organizado en tres fases secuenciales, estructuradas internamente en **capas de menor a mayor dependencia**:

### Fase 1: Datos, ETL y Calidad del Corpus (`scripts/`, `data/`)
Construye el corpus oficial de 1.500 párrafos balanceados (`fiziko`, `biologio`, `kemio`) a partir de Vikipedio y garantiza su integridad antes de entrenar.

| Capa | Qué contiene | Archivos principales |
| :--- | :--- | :--- |
| **Capa 1 · Contrato y limpieza** | Saneamiento de párrafos (HTML, URLs, wikimarkup, `;`), filtro de prosa (`alpha_ratio >= 0.60`) y validación estricta UTF-8 `text;class`. | `src/esperanto_nlp/cleaning.py`, `src/esperanto_nlp/contract.py` |
| **Capa 2 · Extracción ETL** | Configuración de categorías/profundidades (`class_depth`), inspección de subcategorías (`subcats`), estimación (`explore`), construcción (`build`) con caché en `data/raw/` y validación CLI. | `scripts/dataset_config.json`, `scripts/build_dataset.py`, `scripts/validate_dataset.py` |
| **Capa 3 · Auditoría de ruido** | Estimación automática de ruido por ausencia de palabras clave y muestreo estratificado para anotación humana (`--sample`, `--annotate`, `--analyze`). | `scripts/noise_report.py`, `data/noise_sample.csv` |
| **Capa 4 · Artefactos y pruebas** | Corpus generado, metadatos con `article_id` y `text_sha256`, estadísticas de generación y pruebas sin red contra MediaWiki simulada. | `data/dataset.csv`, `data/dataset_meta.csv`, `data/dataset_stats.json`, `tests/test_cleaning_contract.py`, `tests/test_build_dataset.py`, `tests/test_noise_report.py` |

### Fase 2: Librería de NLP en Esperanto (`src/esperanto_nlp/`)
Implementa el pipeline lingüístico modular y explicable en español.

| Capa | Qué contiene | Archivos principales |
| :--- | :--- | :--- |
| **Capa 1 · Primitivas lingüísticas** | Normalización Unicode NFC y sistema-x (`cx` $\rightarrow$ `ĉ`), léxico de palabras cerradas (pronombres, preposiciones, 45 correlativos, numerales) y descomposición derivativa de afijos. | `normalization.py`, `lexicon.py`, `_derivation.py` |
| **Capa 2 · Motor morfológico** | Árbol de decisión único (`analyze()`) con caché `@lru_cache`: léxico cerrado primero, siglas, nombres propios, participios y terminaciones regulares. | `_morph_analyzer.py` |
| **Capa 3 · Contenedores** | Modelos de datos estilo spaCy (`Doc`, `Token`, `Span`), acceso por slices, oraciones (`sents`) y exportación tabular (`to_list`, `to_dicts`). | `containers.py` |
| **Capa 4 · Componentes del pipeline** | Tokenizador con offsets exactos, etiquetador POS, lematizador y gestor de palabras vacías derivadas de la gramática. | `tokenizer.py`, `tagger.py`, `lemmatizer.py`, `stopwords.py`, `resources/stopwords.json` |
| **Capa 5 · Orquestación y explicabilidad** | Clase `EsperantoNLP` (`load()`, `pipe()`, `add_pipe()`, `remove_pipe()`), glosario/explicador morfológico en español, fachada del paquete y CLI interactiva. | `pipeline.py`, `glosario_es.py`, `__init__.py`, `scripts/probar_frase.py` |

### Fase 3: Aprendizaje Automático y Clasificación Web (`notebooks/`, `src/`, `scripts/`)
Conecta la librería lingüística con `scikit-learn` para entrenar y evaluar clasificadores temáticos sin fuga de datos.

| Capa | Qué contiene | Archivos principales |
| :--- | :--- | :--- |
| **Capa 1 · Puente con `scikit-learn`** | Transformador `EsperantoPreprocessor` (`BaseEstimator`, `TransformerMixin`) con caché sin estado compatible con `Pipeline`, `clone` y `GridSearchCV`. | `src/esperanto_nlp/sklearn_wrapper.py`, `tests/test_sklearn_wrapper.py` |
| **Capa 2 · Experimentación y web** | Partición agrupada por artículo (`StratifiedGroupKFold`), comparativa de representaciones (`text`, `norm`, `lemma`, `stem`, `root`) y clasificación de páginas web externas. | `notebooks/clasificacion_esperanto.ipynb`, `data/fallback_web_text.txt` |
| **Capa 3 · Sincronización e integridad** | Empaquetado de respaldo para ejecución autónoma en Colab, ejecución automatizada sin red (`--offline`) y pruebas de regresión sobre metadatos y CV. | `scripts/sync_notebook_snapshot.py`, `scripts/run_notebook.py`, `tests/test_notebook_integrity.py` |

---

## Navegación de la documentación

* **[Guía rápida](guia-rapida.md):** Instalación, primeros pasos en Python, uso por consola y ejemplo con `scikit-learn`.
* **[Gramática implementada](gramatica.md):** Reglas morfológicas, terminaciones, correlativos, casos trampa y afijos reconocidos.
* **[Dataset](dataset.md):** Flujo de construcción ETL (`subcats`, `explore`, `build`), contrato de datos, medición de ruido y trazabilidad.
* **[Referencia de la API](api.md):** Documentación técnica generada automáticamente desde los docstrings del código fuente.