# Decisiones de diseño

Este documento recoge las principales decisiones técnicas de arquitectura e implementación tomadas a lo largo del proyecto, organizadas por **fases y capas**. Para cada decisión se detalla **qué se eligió, por qué y qué alternativa se descartó**.

---

## Fase 1 · Datos, ETL y Calidad del Corpus

### Capa 1 · Contrato y limpieza (`cleaning.py`, `contract.py`)

#### 1. Limpieza compartida entre ETL e inferencia (evitar *train/serve skew*)
* **Qué se eligió:** Separar la descarga de Wikipedia (en `scripts/build_dataset.py`) de la librería principal, pero mantener la limpieza de párrafos dentro del paquete instalable (`esperanto_nlp.cleaning`).
* **Por qué:** Tanto la construcción del dataset como la clasificación posterior de páginas web externas deben aplicar exactamente las mismas reglas de saneamiento (HTML, URLs, wikimarkup y `;` sustituidos por `,`), evitando discrepancias entre entrenamiento e inferencia (*train/serve skew*).
* **Alternativa descartada:** Duplicar las expresiones regulares de limpieza en `scripts/` y en el cuaderno, o meter `requests` dentro de la librería base.

#### 2. Filtro de calidad y ratio alfabético (`min_alpha_ratio = 0.6`)
* **Qué se eligió:** En `is_valid_paragraph` se exige una longitud válida (120–1.500 caracteres, $\ge 15$ palabras, puntuación terminal) y que al menos el 60 % de los caracteres sean letras alfabéticas (`sum(c.isalpha()) / len(text) >= 0.6`).
* **Por qué:** La prosa científica pura ronda el 85 % de caracteres alfabéticos y con fechas o constantes físicas el 78 %, mientras que las fórmulas estequiométricas, ecuaciones o tablas numéricas caen por debajo del 32 %. El umbral `0.60` descarta ruido no lingüístico sin perder párrafos científicos legítimos.
* **Alternativa descartada:** Filtrar solo por número de caracteres o palabras, lo que dejaba pasar tablas y fórmulas densas como texto válido.

#### 3. Contrato estricto UTF-8 sin `;` internos (`contract.py`)
* **Qué se eligió:** Exigir UTF-8 estricto sin BOM, cabecera exacta `text;class`, un párrafo por fila y prohibición absoluta de `;`, HTML, enlaces o corchetes de citas dentro del texto, validado en `scripts/validate_dataset.py` y en CI.
* **Por qué:** Un `;` dentro de un párrafo rompe la lectura columnar del CSV en `pandas` o `csv.reader`, y los restos de marcado contaminan la tokenización y el vocabulario del clasificador.
* **Alternativa descartada:** Escapar comillas o `;` en el CSV y delegar toda la detección de errores al momento de entrenar.

---

### Capa 2 · Extracción ETL (`build_dataset.py`, `dataset_config.json`)

#### 4. Poda manual de subcategorías y profundidad asimétrica (`class_depth`)
* **Qué se eligió:** Inspeccionar el grafo de Vikipedio con `build_dataset.py subcats`, excluir ramas ruidosas en `exclude_subcategories` (biografías como `Fizikistoj`/`Biologoj`, esbozos `Ĝermoj pri...`, ficción/premios) y definir profundidades por clase (`{"fiziko": 1, "biologio": 1, "kemio": 2}`).
* **Por qué:** Con profundidad 2, física arrastraba artículos ajenos como *«Lerneja paŭzo»* (recreo escolar); química, en cambio, tiene menos artículos directos tras excluir esbozos y necesita bajar a nivel 2 para reunir los 500 párrafos objetivo. El flag `--depth N` permite experimentar sin editar el JSON.
* **Alternativa descartada:** Descargar con profundidad fija global sin podar subcategorías, lo que llenaba el corpus de biografías y esbozos de 1 línea.

#### 5. Desambiguación cruzada, tope por artículo y caché local (`data/raw/`)
* **Qué se eligió:** Eliminar cualquier artículo presente en más de una categoría (`drop_ambiguous`), limitar a un máximo de 6 párrafos por artículo (`max_paragraphs_per_article = 6`), balancear por submuestreo determinista (`seed = 42`) y guardar las respuestas JSON crudas en `data/raw/`.
* **Por qué:** Evita artículos ambiguos entre disciplinas, impide que un único artículo largo domine toda una clase y permite reconstruir el corpus de forma reproducible sin repetir peticiones HTTP a Wikipedia.
* **Alternativa descartada:** Tomar todos los párrafos disponibles por clase sin caché ni límite por artículo.

#### 6. Blindaje de codificación multiplataforma en Windows
* **Qué se eligió:** Detectar si `sys.stdout`/`sys.stderr` no son terminales interactivas (`not _stream.isatty()`) y reconfigurarlos a `UTF-8` con `errors="replace"` al inicio de los scripts CLI.
* **Por qué:** En Windows, al redirigir la salida a un archivo (`> log.txt`) o tubería, Python recurre por defecto a `cp1252`, provocando errores `UnicodeEncodeError` con los caracteres del Esperanto (`ĉ, ĝ, ĥ, ĵ, ŝ, ŭ`).
* **Alternativa descartada:** Depender de que el usuario configure manualmente `PYTHONUTF8=1` en cada terminal.

---

### Capa 3 · Auditoría de ruido (`noise_report.py`)

#### 7. Doble auditoría de ruido: indicador automático y muestra manual atómica
* **Qué se eligió:** Combinar un indicador automático (párrafos que no contienen ninguna de las 30 palabras clave discriminantes de su clase) con una muestra estratificada reproducible (`--sample 50 --seed 42`) anotable por consola (`--annotate`) mediante escritura atómica (`tempfile` + `os.replace`).
* **Por qué:** El indicador automático estima la especificidad léxica sin esfuerzo humano, pero no equivale a la tasa real de errores; la muestra manual permite medir qué porcentaje de párrafos son genéricos (fechas de fundación, sedes) o de otra temática sin riesgo de corromper `noise_sample.csv` si se interrumpe la sesión.
* **Alternativa descartada:** Confiar únicamente en métricas automáticas de palabras clave o editar el archivo directamente sin escritura atómica.

---

### Capa 4 · Trazabilidad y pruebas (`dataset_meta.csv`, `tests/`)

#### 8. Pruebas ETL sin red mediante sesión MediaWiki simulada (`FakeSession`)
* **Qué se eligió:** Simular en `tests/test_build_dataset.py` las respuestas de la API MediaWiki (paginación con `cmcontinue`, `categoryinfo` y extractos) mediante inyección de una sesión falsa en memoria.
* **Por qué:** Las pruebas unitarias deben ser deterministas, rápidas y ejecutables en cualquier entorno (incluido CI sin acceso garantizado a Wikipedia).
* **Alternativa descartada:** Realizar peticiones HTTP reales contra `eo.wikipedia.org` durante `pytest`.

---

## Fase 2 · Librería de NLP basada en reglas (`src/esperanto_nlp/`)

### Capa 1 · Primitivas lingüísticas (`normalization.py`, `lexicon.py`, `_derivation.py`)

#### 9. Normalización Unicode NFC y sistema-x prudente con `ux`
* **Qué se eligió:** Unificar primero a forma Unicode NFC y convertir dígrafos `cx, gx, hx, jx, sx` conservando mayúsculas, pero convertir `ux` $\rightarrow$ `ŭ` **solo tras vocal** (`(?<=[aeiouAEIOU])`).
* **Por qué:** Una letra acentuada puede venir descompuesta (`c` + acento combinante) o escrita en sistema-x; restringir `ux` tras vocal evita corromper préstamos o nombres propios extranjeros como `lux` o `Luxemburgo`.
* **Alternativa descartada:** Reemplazar `ux` $\rightarrow$ `ŭ` de forma incondicional en todo el texto.

#### 10. Léxico cerrado generado y consulta prioritaria antes que terminaciones
* **Qué se eligió:** Consultar `LEXICON` antes de aplicar las reglas de terminación (`-o`, `-a`, `-e`, etc.) y generar programáticamente las 45 bases de correlativos (*tabelvortoj*: 5 prefijos $\times$ 9 sufijos) junto con sus flexiones `-j`/`-n` y numerales compuestos (`is_compound_numeral` con programación dinámica).
* **Por qué:** Palabras muy frecuentes como `kio` (acaba en `-o`, pero es `PRON`) o `tre` (acaba en `-e`, pero es `ADV` sin raíz `tr-`) quedarían mal etiquetadas si se evaluaran primero las terminaciones regulares.
* **Alternativa descartada:** Escribir a mano cientos de entradas flexionadas o aplicar terminaciones antes del léxico.

#### 11. Descomposición derivativa conservadora de afijos (`_derivation.py`)
* **Qué se eligió:** Separar únicamente prefijos/sufijos de alta precisión (`mal-` y sufijos `SUFFIXES`), exigiendo raíz residual $\ge 3$ letras con al menos una vocal, protegiendo falsos positivos en `NO_SPLIT` (`ekzist`, `koleg`, `malari`...) y exponiendo el resultado solo en `token.root_` y `token.affixes` (sin alterar `token.lemma_`).
* **Por qué:** Sin un diccionario completo de raíces, sufijos ambiguos como `-ul`, `-ar`, `-ad`, `-id`, `-il`, `-et`, `-em`, `-an` o `-in` (salvo `-in` tras `-ist`) romperían vocabulario científico real (`molekulo`, `polara`, `oksido`, `planeto`, `sistemo`, `vitamino`).

| Atributo | Ejemplo con `lernejoj` | Qué elimina |
| :--- | :--- | :--- |
| `lemma_` | `lernejo` | Solo flexión `-j`/`-n` (forma canónica de diccionario) |
| `stem_` | `lernej` | La terminación gramatical (`-o`, `-a`, `-e`, `-as`...) |
| `root_` | `lern` | Además, afijos derivativos seguros |

* **Alternativa descartada:** Recortar todos los afijos gramaticales del Esperanto de forma agresiva, lo que mutilaría raíces científicas técnicas.

---

### Capa 2 · Motor morfológico central (`_morph_analyzer.py`)

#### 12. Un único analizador morfológico interno con caché LRU (`maxsize=200_000`)
* **Qué se eligió:** Centralizar todo el árbol de decisión morfológica en `_morph_analyzer.analyze()` (devolviendo un `MorphAnalysis` inmutable con `MappingProxyType`) decorado con `@lru_cache(maxsize=200_000)`.
* **Por qué:** Tanto `Tagger` como `Lemmatizer` llaman a `analyze()` sobre los mismos tokens; centralizar evita reglas contradictorias y el caché LRU hace que la segunda llamada —y cualquier palabra repetida en el corpus— se resuelva en $O(1)$ sin volver a evaluar expresiones regulares.
* **Alternativa descartada:** Mantener reglas separadas en el etiquetador y el lematizador o recalcular cada token sin caché.

#### 13. Lema de los participios = infinitivo
* **Qué se eligió:** Lematizar los participios activos/pasivos (`-ant-`, `-int-`, `-ont-`, `-at-`, `-it-`, `-ot-`) al infinitivo del verbo (`kuranta` $\rightarrow$ `kuri`), conservando el rasgo en `morph` (`VerbForm=Part`, `Voice=Act/Pass`, `Tense=...`).
* **Por qué:** Reduce la dispersión de vocabulario para el clasificador agrupando formas verbales y participiales bajo un mismo lema.
* **Alternativa descartada:** Dejar `kuranta` con lema `kuranto` / `kuranta`, lo que fragmentaría el conteo de frecuencias.

---

### Capa 3 · Contenedores (`containers.py`)

#### 14. Texto original intacto y desplazamiento exacto en `Token`, `Span` y `Doc`
* **Qué se eligió:** Mantener `token.text` y `token.idx` apuntando literalmente al texto de entrada, guardando la versión normalizada en `token.norm_`.
* **Por qué:** Garantiza la invariante `doc.text[t.idx:t.end_idx] == t.text` y que `Span.text` reconstruya exactamente el fragmento original sin descuadres de índices.
* **Alternativa descartada:** Normalizar el texto antes de tokenizar, lo que rompería la correspondencia con los índices originales.

---

### Capa 4 · Componentes del pipeline (`tokenizer.py`, `tagger.py`, `lemmatizer.py`, `stopwords.py`)

#### 15. Tokenizador por prioridades y stop-words derivadas de la gramática
* **Qué se eligió:** Tokenizar en 5 grupos ordenados (`url` $\rightarrow$ `abbr` $\rightarrow$ `number` $\rightarrow$ `word` $\rightarrow$ `punct`) respetando iniciales y abreviaturas (`L. L. Zamenhof`, `k.t.p.`), y construir `StopWords` por defecto desde las palabras cerradas funcionales de `lexicon.py` más las formas de `esti` (`DEFAULT_STOPWORDS`).
* **Por qué:** El orden de patrones impide partir URLs, correos o abreviaturas con punto; derivar las stop-words de la gramática evita eliminar términos frecuentes propios del dominio científico (algo que ocurriría si se calcularan por frecuencia sobre el dataset).
* **Alternativa descartada:** Tokenizar con `str.split()` o extraer stop-words por TF/frecuencia del corpus.

---

### Capa 5 · Orquestación y explicabilidad (`pipeline.py`, `glosario_es.py`, `probar_frase.py`)

#### 16. Patrón *pipeline* modular y capa auxiliar de explicabilidad en español
* **Qué se eligió:** Encadenar `Tokenizer -> Tagger -> Lemmatizer -> StopWords -> Doc` en `EsperantoNLP` (con `add_pipe`, `remove_pipe` y el generador `pipe()`), y mantener `glosario_es.py` completamente desacoplado del análisis y del modelo.
* **Por qué:** Cada componente tiene una única responsabilidad y puede probarse aislado; el glosario en español facilita interpretar salidas morfológicas y términos importantes del clasificador sin introducir sesgos en las predicciones.
* **Alternativa descartada:** Implementar una función monolítica `procesar(texto)` difícil de depurar y extender.

---

## Fase 3 · Aprendizaje Automático y Clasificación Web

### Capa 1 · Puente con `scikit-learn` (`sklearn_wrapper.py`)

#### 17. Transformador compatible con `Pipeline` y caché sin estado
* **Qué se eligió:** Implementar `EsperantoPreprocessor` heredando de `BaseEstimator` y `TransformerMixin`, delegando en una función global cacheada (`@lru_cache(maxsize=200_000)`) en lugar de guardar la instancia `nlp` dentro del estimador.
* **Por qué:** Permite usar `clone()`, `GridSearchCV`, `cross_val_score` y serializar con `pickle` sin arrastrar el objeto del pipeline; además, como el preprocesado lingüístico es determinista y no aprende parámetros de los datos, cachearlo evita repetir todo el análisis NLP en cada *fold* de la validación cruzada sin producir *data leakage*.
* **Alternativa descartada:** Preprocesar una columna fija fuera del `Pipeline` de `scikit-learn` o almacenar `EsperantoNLP` como atributo serializable del estimador.

---

### Capa 2 · Experimentación, validación y clasificación web (`clasificacion_esperanto.ipynb`)

#### 18. Partición agrupada por artículo (`StratifiedGroupKFold`) y ablación de representaciones
* **Qué se eligió:** Usar `dataset_meta.csv` (`article_id` y verificación criptográfica `text_sha256`) para dividir entrenamiento/test y validación cruzada con `StratifiedGroupKFold`, comparando las representaciones `text`, `norm`, `lemma`, `stem` y `root` en igualdad de condiciones.
* **Por qué:** Párrafos procedentes de un mismo artículo de Wikipedia comparten vocabulario específico y nombres propios; si se repartieran aleatoriamente entre entrenamiento y evaluación, las métricas saldrían infladas artificialmente (*data leakage*).
* **Alternativa descartada:** Utilizar `train_test_split` o `StratifiedKFold` plano por filas ignorando el `article_id`.

---

### Capa 3 · Sincronización, ejecución automática e integridad (`sync_notebook_snapshot.py`, `run_notebook.py`, `test_notebook_integrity.py`)

#### 19. Respaldo embebido verificable y ejecución offline determinista
* **Qué se eligió:** Empaquetar el código fuente de `src/esperanto_nlp/*.py` en `EMBEDDED_LIBRARY_B64` (con su `EMBEDDED_LIBRARY_SHA256`) dentro del cuaderno mediante `sync_notebook_snapshot.py`, validarlo automáticamente en `test_notebook_integrity.py`, y automatizar el cuaderno sin red mediante `run_notebook.py --offline` (usando `data/fallback_web_text.txt`).
* **Por qué:** Garantiza que el cuaderno pueda arrancar de forma autónoma en entornos como Google Colab y que la integración continua (`.github/workflows/ci.yml`) verifique tanto las pruebas unitarias como la ejecución completa del cuaderno sin depender de la red.
* **Alternativa descartada:** Dejar una copia manual desincronizada dentro del cuaderno o depender exclusivamente de conexiones externas para validar el proyecto.

---

## Limitaciones conocidas (posibles líneas de mejora)
* Sin un diccionario completo de raíces, algunas palabras no listadas pueden imitar falsos participios (mitigado con `_NON_PARTICIPLE_STEMS` y longitud mínima de raíz).
* La descomposición de afijos es heurística y excluye afijos ambiguos frecuentes en ciencia; podría ampliarse validando raíces candidatas frente a frecuencias del propio corpus.
* No se descomponen palabras compuestas por múltiples raíces léxicas (p. ej. `akvofalo` = `akv` + `fal`).
* Las mayúsculas en mitad de frase se asumen nombres propios salvo titulares largos en mayúsculas (puede fallar tras `:` o en citas).
* Ciertas ambigüedades categoriales cerradas (`kiu` como `PRON` frente a `DET`) se resuelven siempre con la etiqueta principal del léxico sin desambiguación sintáctica por contexto.
