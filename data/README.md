# Directorio de Datos y Trazabilidad del Corpus (`data/`)

Este directorio almacena el corpus oficial en Esperanto extraído de [Vikipedio](https://eo.wikipedia.org), junto con sus metadatos de procedencia, estadísticas de construcción y muestras de auditoría de calidad.

---

## 1. Inventario de Archivos

| Archivo / Ruta | Descripción y Propósito |
| :--- | :--- |
| **`dataset.csv`** | **Corpus principal (Contrato de Datos).** Contiene **1.500 párrafos** reales y balanceados (**500** de `fiziko`, **500** de `biologio` y **500** de `kemio`) procedentes de **464 artículos** únicos. Codificado en UTF-8 (sin BOM), delimitado por `;` y con cabecera `text;class`. |
| **`dataset_meta.csv`** | **Metadatos de trazabilidad fila a fila.** Registra para cada fila de `dataset.csv` su índice (`row`), clase (`class`), identificador de página (`article_id`), título (`article_title`), ID de revisión exacta en Wikipedia (`revid`) y huella criptográfica (`text_sha256`). Esencial para agrupar por artículo (`StratifiedGroupKFold`) y evitar fuga de datos (*data leakage*) entre entrenamiento y test. |
| **`dataset_stats.json`** | **Registro de reproducibilidad.** Documenta la fecha de extracción UTC, la semilla (`seed = 42`), el número de artículos por clase (`fiziko`: 136, `biologio`: 172, `kemio`: 156) y la configuración exacta de `scripts/dataset_config.json` empleada en su generación. |
| **`noise_sample.csv`** | **Muestra para auditoría de ruido.** Contiene **150 párrafos** (50 por clase) muestreados de forma estratificada para evaluar el ruido por herencia de etiqueta (`0` = informativo de su clase, `1` = genérico/biográfico, `2` = otra temática). |
| **`fallback_web_text.txt`** | **Texto de respaldo para inferencia web.** Párrafos de reserva en Esperanto que utiliza el cuaderno en la sección de clasificación de páginas web externas (Fase 3) cuando se ejecuta sin conexión (`OFFLINE=True`) o si las fuentes web externas no están disponibles. |
| **`raw/`** | **Caché local de la API MediaWiki.** Almacena los JSON crudos descargados por `pageid` para evitar peticiones repetidas a Wikipedia. Está excluida del control de versiones en `.gitignore`. |

---

## 2. Garantías del Contrato de Datos

El archivo `dataset.csv` cumple todas las restricciones verificadas por `esperanto_nlp.contract`:
- **Formato:** CSV en UTF-8 con separador `;` y cabecera estricta `text;class`.
- **Limpieza:** Texto libre de etiquetas HTML, enlaces HTTP/HTTPS, plantillas `{{...}}`, referencias `[1]`, saltos de línea internos y caracteres `;` dentro del texto.
- **Filtro de prosa:** Párrafos de entre 120 y 1.500 caracteres, mínimo 15 palabras, puntuación terminal válida y ratio alfabético $\ge 0.60$.
- **Desambiguación:** Cualquier artículo que pertenezca simultáneamente a más de una categoría temática se descarta antes de extraer sus párrafos.

---

## 3. Comandos de Validación, Auditoría y Reconstrucción

No es necesario volver a descargar el corpus desde Wikipedia para ejecutar las pruebas ni el cuaderno. Si deseas validar o auditar los archivos existentes, ejecuta desde la raíz del repositorio:

```bash
# 1. Validar que dataset.csv cumple el contrato de datos
python scripts/validate_dataset.py data/dataset.csv

# 2. Calcular la estimación automática de ruido por ausencia de palabras clave
python scripts/noise_report.py

# 3. Anotar interactivamente la muestra manual o consultar sus métricas
python scripts/noise_report.py --annotate
python scripts/noise_report.py --analyze
```

> **Nota sobre reconstrucción:** Si deseas probar el proceso de descarga completa desde cero sin sobrescribir el corpus oficial ni perder anotaciones en `noise_sample.csv`, especifica un directorio de salida alternativo:
> ```bash
> python scripts/build_dataset.py build --out data_nueva
> ```

---

## 4. Uso en Local y en Google Colab

- **Ejecución local:** El cuaderno `notebooks/clasificacion_esperanto.ipynb` detecta y carga automáticamente `data/dataset.csv`, `data/dataset_meta.csv` y `data/fallback_web_text.txt` desde este directorio.
- **Ejecución en Google Colab:** Cuando el cuaderno se abre en la nube sin clonado previo, descarga automáticamente estos archivos desde el repositorio de GitHub para garantizar que los resultados sean 100 % reproducibles.
