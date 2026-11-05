# Construcción, Validación y Auditoría del Corpus (`data/`)

Esta sección documenta el ciclo de vida completo del dataset oficial del proyecto: desde el rastreo de categorías en la Wikipedia en Esperanto ([Vikipedio](https://eo.wikipedia.org)) hasta su extracción ETL, limpieza rigurosa, balanceo, validación del contrato de datos y auditoría del ruido de etiquetas.

---

## 1. Flujo de Trabajo y Comandos CLI

El proceso de creación y control de calidad no es una descarga directa a ciegas, sino un proceso en tres fases consecutivas:

![alt text](image.png)

El orden de ejecución de comandos en la terminal es el siguiente:

```bash
# --- FASE 1: Inspección previa ---
# 1. Inspeccionar el árbol de subcategorías para detectar fuentes de ruido
python scripts/build_dataset.py subcats

# 2. Evaluar si el volumen de artículos y párrafos es suficiente
python scripts/build_dataset.py explore

# (Opcional) Probar con distinta profundidad sin editar la configuración
python scripts/build_dataset.py explore --depth 1

# --- FASE 2: Extracción ETL ---
# 3. Ejecutar el pipeline ETL completo (descarga, limpieza, balanceo)
python scripts/build_dataset.py build

# --- FASE 3: Validación y Auditoría ---
# 4. Validar que el CSV cumple estrictamente el contrato de datos
python scripts/validate_dataset.py data/dataset.csv

# 5. Auditar el ruido de etiquetas automático y generar muestra manual
python scripts/noise_report.py
python scripts/noise_report.py --sample 50
python scripts/noise_report.py --annotate --class fiziko
python scripts/noise_report.py --analyze
```

---

## 2. Guía Detallada de Comandos de Construcción (`build_dataset.py`)

El script `scripts/build_dataset.py` es el orquestador ETL. Implementa tres subcomandos esenciales:

### A. Subcomando `subcats` (Inspección y Poda de Ruido)
* **Propósito:** Lista exhaustivamente todas las subcategorías encontradas bajo las categorías raíz de cada clase, mostrando cuántos artículos y cuántas subcategorías anidadas contiene cada una.
* **Uso:**
  ```bash
  python scripts/build_dataset.py subcats
  python scripts/build_dataset.py subcats --config scripts/dataset_config.json
  ```
* **Por qué es crítico:** Wikipedia estructura sus categorías como grafos que mezclan conceptos teóricos con personas, premios y cultura. Gracias a `subcats`, se detectaron ramas que introducían texto no científico:
  - *Biografías de científicos:* `Fizikistoj`, `Biologoj`, `Kemiistoj`, `Nobelpremiitoj` (generan ruido biográfico: fechas de nacimiento, lugares de estudio y cargos académicos).
  - *Esbozos:* `Ĝermoj pri fiziko`, `Ĝermoj pri biologio`, `Ĝermoj pri kemio` (artículos de una o dos líneas sin valor semántico).
  - *Cultura y divulgación no científica:* `Fiziko en fikcio`, `Fizikaj premioj`, `Magio`, `Bioetiko`, `Kemio en socio`.
  - Todas estas ramas se incorporaron a la lista `exclude_subcategories` de `dataset_config.json` para podarlas antes de la extracción.

---

### B. Subcomando `explore` (Estimación de Volumen sin Descargas Masivas)
* **Propósito:** Recorre el árbol de categorías respetando las exclusiones y calcula el número de artículos candidatos disponibles y una estimación de los párrafos que se podrían extraer.
* **Uso:**
  ```bash
  python scripts/build_dataset.py explore
  python scripts/build_dataset.py explore --depth 1     # Probar profundidad 1 en todas las clases
  python scripts/build_dataset.py explore --verbose     # Ver el desglose detallado artículo por artículo
  ```
* **Por qué es crítico:** Permite saber con certeza si alcanzaremos el objetivo de párrafos por clase (p. ej. 500 párrafos) **antes** de lanzar la descarga pesada de los cuerpos de los artículos, ahorrando ancho de banda y tiempo.

---

### C. Subcomando `build` (Ejecución Completa del Pipeline ETL)
* **Propósito:** Ejecuta las tres capas del pipeline de datos:
  1. **Extract:** Descarga los artículos candidatos mediante la API MediaWiki (`requests.Session` con cabecera `User-Agent` descriptiva y pausas de cortesía de 0.2 segundos). Guarda cada respuesta JSON cruda en `data/raw/` (caché local).
  2. **Transform:** Descarta artículos ambiguos (que aparezcan en más de una clase). Extrae los párrafos con `extract_paragraphs`, elimina cabeceras (`== ... ==`), amputa referencias (`== Referencoj ==`), limpia artefactos con `esperanto_nlp.cleaning.clean_paragraph`, y aplica filtros de longitud (120–1.500 caracteres, $\ge 15$ palabras) y ratio alfabético ($\ge 0.60$).
  3. **Load:** Selecciona un máximo de 6 párrafos por artículo para evitar que un único artículo extenso domine una clase, submuestrea de forma reproducible (`seed = 42`) para balancear todas las clases exactamente al objetivo (500 párrafos por clase) y genera los archivos de salida.
* **Uso:**
  ```bash
  # Construcción oficial del corpus
  python scripts/build_dataset.py build

  # Construcción en un directorio alternativo (para pruebas sin sobrescribir el corpus oficial)
  python scripts/build_dataset.py build --out data_prueba

  # Forzar una profundidad global de rastreo distinta
  python scripts/build_dataset.py build --depth 1 --out data_depth1
  ```
* **Salidas generadas:**
  - `dataset.csv`: Corpus limpio con formato `text;class`.
  - `dataset_meta.csv`: Trazabilidad fila a fila (`row`, `class`, `article_id`, `article_title`, `revid`, `text_sha256`).
  - `dataset_stats.json`: Parámetros exactos de configuración, semilla y recuentos finales.

---

## 3. Configuración del Dataset (`dataset_config.json`)

El archivo `scripts/dataset_config.json` parametriza todo el proceso:

```json
{
  "classes": {
    "fiziko": ["Fiziko"],
    "biologio": ["Biologio"],
    "kemio": ["Kemio"]
  },
  "class_depth": {
    "fiziko": 1,
    "biologio": 1,
    "kemio": 2
  },
  "target_paragraphs_per_class": 500,
  "max_paragraphs_per_article": 6,
  "min_chars": 120,
  "max_chars": 1500,
  "min_words": 15,
  "seed": 42,
  "balance_classes": true
}
```

### Decisión de Diseño: Profundidad Asimétrica por Clase (`class_depth`)
- `max_category_depth` define la profundidad por defecto (2 niveles).
- `class_depth` permite ajustar la profundidad clase por clase:
  - **`fiziko`: profundidad 1.** Con profundidad 2, la categoría arrastraba subcategorías indirectas de educación y vida escolar (p. ej. el artículo *«Lerneja paŭzo»* —recreo escolar— apareció clasificado erróneamente bajo física). Al limitar a 1 nivel, el corpus queda limpio y focalizado en conceptos físicos puros.
  - **`biologio`: profundidad 1.** Biología cuenta con abundantes artículos directos sobre taxonomía y fisiología, por lo que 1 nivel aporta más que suficiente volumen.
  - **`kemio`: profundidad 2.** Química en Esperanto tiene menos artículos raíz una vez excluidos los esbozos; requiere descender un segundo nivel para reunir los 500 párrafos objetivo sin comprometer la pureza temática.
- El flag `--depth N` de la línea de comandos permite anular temporalmente esta configuración para hacer pruebas rápidas de exploración.

---

## 4. Validación del Contrato de Datos (`validate_dataset.py`)

El script `scripts/validate_dataset.py` verifica formalmente el cumplimiento del contrato de datos del proyecto:

```bash
python scripts/validate_dataset.py data/dataset.csv
```

### Reglas verificadas por `esperanto_nlp.contract`:
1. **Codificación:** UTF-8 estricto sin marca BOM inicial.
2. **Cabecera:** Exactamente `text;class` en la primera línea.
3. **Delimitador:** Delimitado exclusivamente por `;`. El texto de los párrafos **nunca** contiene `;` (fueron convertidos en comas o eliminados durante la fase de limpieza).
4. **Ausencia de artefactos web:** Cero etiquetas HTML (`<p>`, `<b>`), cero URLs o hiperenlaces (`http://`, `https://`), cero plantillas MediaWiki (`{{...}}`) y cero corchetes de citas o referencias (`[1]`, `[citaĵo bezonata]`).
5. **Calidad de prosa:** Ratio alfabético $\ge 0.60$ (descarta fórmulas aisladas y tablas de datos) y longitud admisible.
6. **Balance:** Verifica que existan al menos 2 clases y un mínimo de filas por clase.

---

## 5. Medición y Auditoría del Ruido de Etiquetas (`noise_report.py`)

Un reto fundamental en la recolección supervisada de corpus enciclopédicos es el **ruido por herencia de etiqueta**. Un párrafo como *«Ĝi estis fondita en 1950.»* dentro del artículo *«Fizika Instituto»* hereda formalmente la etiqueta `fiziko`, pero no contiene contenido físico explicativo.

El script `scripts/noise_report.py` ofrece dos enfoques complementarios:

### A. Indicador Automático por Palabras Clave
Calcula las 30 palabras más características de cada clase (mediante ratio de frecuencia contrastada contra las demás) y cuenta qué porcentaje de párrafos de una clase **no contienen ni una sola** de sus palabras clave:

```bash
python scripts/noise_report.py --data data/dataset.csv
```

*Interpretación:* Es un indicador automático y exploratorio; la ausencia de palabras clave no implica necesariamente que un párrafo sea incorrecto, pero sirve como termómetro de especificidad léxica.

---

### B. Muestreo y Auditoría Humana (`--sample`, `--annotate`, `--analyze`)
Permite generar una muestra representativa para etiquetar manualmente la calidad real del corpus:

1. **Generar la muestra determinista:**
   ```bash
   python scripts/noise_report.py --sample 50 --out data/noise_sample.csv
   ```
   Extrae 50 párrafos al azar por cada clase usando `seed = 42` (150 filas en total) con la columna `generic` en blanco. Protege contra sobrescrituras accidentales.

2. **Sesión interactiva de etiquetado:**
   ```bash
   python scripts/noise_report.py --annotate
   python scripts/noise_report.py --annotate --class fiziko   # Filtrar solo una clase
   ```
   Presenta los párrafos pendientes por consola permitiendo asignar:
   - `0`: Párrafo **informativo** y característico de la ciencia en cuestión.
   - `1`: Párrafo **genérico / ruido** (fechas de fundación, sedes, biografías generales).
   - `2`: Párrafo perteneciente a **otra temática** no relacionada.
   - `s`: Saltar y dejar pendiente.
   - `q`: Guardar y salir.
   *Nota técnica:* Guarda en disco de forma atómica e inmediata tras cada respuesta; se puede pausar y reanudar en cualquier momento.

3. **Análisis de resultados humanos:**
   ```bash
   python scripts/noise_report.py --analyze
   ```
   Calcula los porcentajes de párrafos informativos, genéricos y temáticamente erróneos sobre el conjunto de párrafos efectivamente anotados, advirtiendo si la muestra está incompleta.

---

## 6. Trazabilidad y Prevención de Fuga de Datos (*Data Leakage*)

En `data/dataset_meta.csv` se registran los metadatos correspondientes fila por fila con `data/dataset.csv`:

| Columna | Significado | Importancia en Modelado |
| :--- | :--- | :--- |
| `row` | Índice de fila (0 a 1.499) | Alineación 1:1 con `dataset.csv`. |
| `class` | Etiqueta temática (`fiziko`, `biologio`, `kemio`) | Verificación de coincidencia de clase. |
| `article_id` | Identificador único de página en Wikipedia | **Agrupación en validación cruzada:** Permite emplear `StratifiedGroupKFold(groups=meta.article_id)` en el cuaderno para garantizar que los párrafos del mismo artículo no se repartan entre entrenamiento y test (*data leakage*). |
| `article_title` | Título del artículo en Wikipedia | Auditoría de procedencia e interpretabilidad. |
| `revid` | ID de revisión exacta de la Wikipedia | Reproducibilidad histórica exacta. |
| `text_sha256` | Hash SHA-256 del texto en UTF-8 | Verificación criptográfica de integridad. Si se altera el texto o el orden de las filas, el cuaderno detecta la discrepancia y rehúsa evaluar. |
