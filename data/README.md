# Datos incluidos en la versión 0.3.1

| Archivo | Contenido |
|---|---|
| dataset.csv | 1.500 párrafos reales; 500 de física, biología y química; formato text;class |
| dataset_meta.csv | Artículo, revisión, clase, índice y SHA-256 del texto de cada fila |
| dataset_stats.json | Estadísticas, semilla y configuración de la generación original |
| noise_sample.csv | 50 párrafos por clase; plantilla de anotación opcional, sin respuestas de prueba |
| fallback_web_text.txt | Texto de demostración si fallan las fuentes web |
| raw/ | Caché de los artículos descargados, incluida en el ZIP para trazabilidad; ignorada por Git |

El cuaderno usa estos datos automáticamente en local. En Colab los descarga desde el repositorio: hay que publicar dataset.csv, dataset_meta.csv y el texto de respaldo en el commit seleccionado por GIT_REF. La publicación remota no se ha realizado como parte de esta revisión.

No hace falta volver a descargar el corpus para verificar la práctica. Para generar otra versión usa `build --out data_nueva`. No sobrescribas la muestra cuando hayas empezado a anotar.
