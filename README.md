# NLP Esperanto - Práctica de Minería de Textos

Sistema integral de Procesamiento de Lenguaje Natural (NLP) y clasificación temática para textos en **Esperanto**, desarrollado para la asignatura de *Minería de Textos* (CUNEF Universidad).

El proyecto consta de tres componentes principales:
1. **Librería de NLP basada en reglas** (`esperanto_nlp`): Construida desde cero (sin dependencias como spaCy o NLTK para el núcleo lingüístico), ofreciendo una API análoga a spaCy.
2. **Generación de Dataset**: Script reproducible de extracción y curación de párrafos desde Wikipedia en Esperanto, conforme al contrato estricto `text;class`.
3. **Cuaderno de Clasificación y Web Scraping**: Cuaderno Jupyter ejecutable de forma autónoma en Google Colab para entrenamiento, validación de modelos, evaluación de rendimiento y clasificación en vivo de páginas web.

---

## 📁 Estructura del Proyecto

```text
nlp_esperanto_practica/
├── .github/workflows/ci.yml       # CI con GitHub Actions (pytest automatizado)
├── docs/                          # Documentación para Read the Docs / MkDocs
├── mkdocs.yml                     # Configuración de MkDocs
├── pyproject.toml                 # Especificación del paquete Python e instalación pip
├── README.md                      # Documentación principal
├── .gitignore                     # Exclusiones de Git
├── data/                          # Datasets conforme al contrato
│   ├── esperanto_dataset.csv      # Dataset principal (text;class)
│   └── sample_test.csv            # Dataset para prueba de independencia temática
├── scripts/
│   └── build_dataset.py           # Script reproducible de recolección de datos
├── src/esperanto_nlp/             # Código fuente de la librería
│   ├── containers.py              # Clases Doc y Token
│   ├── tokenizer.py               # Tokenizador de frases y palabras
│   ├── morpho_rules.py            # Reglas morfológicas del Esperanto
│   ├── tagger.py                  # POS Tagger basado en reglas
│   ├── lemmatizer.py              # Lematizador morfológico
│   ├── stopwords.py               # Manejo de stop-words
│   ├── pipeline.py                # Pipeline orquestador EsperantoNLP
│   └── wiki/                      # Cliente API para MediaWiki / Wikidata
├── tests/                         # Pruebas unitarias (pytest)
└── notebooks/
    └── clasificacion_esperanto.ipynb  # Cuaderno de experimentación y evaluación
```

---

## 🚀 Instalación

La librería está diseñada para ser instalable directamente mediante `pip` tanto en local como en Google Colab:

### Instalación en local (modo desarrollo):
```bash
git clone https://github.com/<tu-usuario>/nlp_esperanto_practica.git
cd nlp_esperanto_practica
pip install -e ".[test,notebook]"
```

### Instalación remota (Google Colab):
```bash
!pip install git+https://github.com/<tu-usuario>/nlp_esperanto_practica.git
```

---

## 💡 Uso Rápido de la Librería

```python
import esperanto_nlp

nlp = esperanto_nlp.load()
doc = nlp("La granda hundo kuris rapide en la ĝardeno.")

for token in doc:
    print(f"{token.text:12} | Lemma: {token.lemma_:10} | POS: {token.pos_:6} | Stop: {token.is_stop}")
```

---

## 📋 Contrato de Datos

Cualquier dataset utilizado en este proyecto debe cumplir las siguientes especificaciones:
- **Formato**: CSV codificado en **UTF-8**.
- **Separador**: Punto y coma (`;`).
- **Columnas**: Exactamente dos columnas con cabecera `text;class`.
- **Granularidad**: Párrafo individual.

---

## 🧪 Ejecución de Tests

```bash
pytest tests/ -v
```
