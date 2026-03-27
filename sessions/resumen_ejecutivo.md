# Resumen ejecutivo de sesiones de trabajo

**Proyecto:** drone-waste-detect-compress
**Autor:** Esp. Ing. Jorge Ceferino Valdez (UNPA)
**Tesis:** Maestría en Informática y Sistemas
**Período:** marzo 2026

---

## Sesión 1 — Auditoría del estado del proyecto y plan de migración

### Objetivo
Leer el CLAUDE.md, recorrer la estructura completa del proyecto y producir un diagnóstico
de qué código del proyecto anterior (cultivos) podía reutilizarse, qué necesitaba adaptarse
y qué debía crearse desde cero en el nuevo dominio (residuos costeros).

### Archivos analizados (lectura, sin modificaciones)
- `CLAUDE.md`
- Todo `src/` (config, utils, models, data)
- Todos los notebooks existentes (0.0 a 12.0 del proyecto de cultivos)
- `requirements.txt`, `setup.py`, `pyproject.toml`

### Decisiones técnicas tomadas

| Categoría | Decisión |
|---|---|
| Reutilizar directo | `src/utils/datasets.py`, `src/utils/metrics.py` (parcial), `validate_batch.py` CompressAI |
| Adaptar | `config.yaml`, `make_dataset.py`, `images.py`, módulo CompressAI completo, notebooks 0.0–6.1 |
| Crear desde cero | `src/utils/throughput.py`, todo `src/models/yolo_model/`, notebooks 7.0–9.0, 11.0 |
| Eliminar/archivar | `convolutional_autoencoder_model/` (CAE del proyecto de cultivos) |
| Renombrar | `compressai_chang2020_model/` → `compressai_cheng2020_model/` (typo en nombre) |

### Hallazgos relevantes
- Los notebooks existentes tienen numeración del proyecto de cultivos (0.0–12.0) y no coinciden con la numeración definida en el CLAUDE.md (0.0–11.0); deben rehacerse.
- `compute_bpp()` existía duplicada en `train_batch.py` y `validate_batch.py`.
- `pad_image_to_multiple()` estaba aislada en su propio archivo `image_processing.py` en lugar de estar en `utils/images.py`.
- El `data/interim/dataset/` contenía imágenes de cultivos (bean, corn, etc.), no de residuos costeros — el dataset real todavía no había sido cargado.

### Pendientes identificados
- Reemplazar los notebooks con la numeración y contenido correctos.
- Capturar el dataset propio con el dron DJI Mini 4 Pro sobre el Golfo San Jorge.
- Configurar X-AnyLabeling para el etiquetado de las 7 clases de residuos.

---

## Sesión 2 — Limpieza y refactoring del código heredado

### Objetivo
Ejecutar las cuatro acciones de limpieza identificadas en la sesión anterior:
renombrar el módulo CompressAI, archivar el CAE, mover `pad_image_to_multiple()` y
mover `compute_bpp()`.

### Archivos modificados
| Archivo | Acción |
|---|---|
| `src/models/compressai_cheng2020_model/` | Creado por renombrado desde `compressai_chang2020_model/` |
| `src/models/compressai_cheng2020_model/train_model.py` | Actualizado import `chang` → `cheng` |
| `src/models/compressai_cheng2020_model/train_batch.py` | Eliminada `compute_bpp()` local; agrado import desde `utils.metrics` |
| `src/models/compressai_cheng2020_model/validate_batch.py` | Eliminada `compute_bpp()` local y `import math`; agregado import desde `utils.metrics` |
| `src/models/compressai_cheng2020_model/image_processing.py` | **Eliminado** |
| `src/utils/images.py` | Agregada `pad_image_to_multiple()` e `import math` |
| `src/utils/metrics.py` | Agregada `compute_bpp()` al inicio del archivo |
| `archive/convolutional_autoencoder_model/` | Creado (movido desde `src/models/`) |
| `src/models/compressai_chang2020_model/` | **Eliminado** |
| `src/models/convolutional_autoencoder_model/` | **Eliminado** de `src/` |

### Decisiones técnicas tomadas
- `compute_bpp()` tenía código **idéntico** en `train_batch.py` y `validate_batch.py`; la fuente canónica queda en `utils/metrics.py` y ambos módulos importan desde allí.
- El CAE se archivó (no eliminó) para preservar la historia del proyecto anterior sin que interfiera con el paquete `src`.
- `pad_image_to_multiple()` se movió a `utils/images.py` porque conceptualmente es una operación de procesamiento de imagen, no una dependencia del modelo.

### Pendientes
- Ninguno específico de esta sesión; los restantes vienen de la sesión 1.

---

## Sesión 3 — Reescritura del sistema de configuración centralizado

### Objetivo
Reescribir `src/config.yaml` y `src/config.py` para el nuevo proyecto, cubriendo
rutas de datos, parámetros de CompressAI para q=1..6, logs por modelo/calidad,
parámetros de entrenamiento YOLO para las tres variantes y parámetros del
experimento de throughput.

### Archivos modificados/creados
| Archivo | Acción |
|---|---|
| `src/config.yaml` | Reescrito completamente (7 secciones, ~180 líneas) |
| `src/config.py` | Reescrito completamente (eliminado `print` en import, nuevos diccionarios y funciones) |

### Estructura de `config.yaml` resultante

| Sección | Contenido clave |
|---|---|
| `data` | Rutas raw/interim/processed/external, 3 datasets públicos, 7 clases, splits con semilla 42 |
| `models` | Pesos `.pth` para cheng2020 q1–q6 y `.pt` para baseline/reconstructed/mixed |
| `compressai` | Arquitectura, calidades [1..6], hiperparámetros de entrenamiento, λ por calidad (0.0018–0.048), parámetros Optuna |
| `yolo` | YOLOv8s, imgsz=640, epochs=100, batch=16, patience=20, AdamW; variantes con rutas propias |
| `logs` | 9 directorios TensorBoard separados (6 CompressAI + 3 YOLO) |
| `throughput` | fps [1,2,5], 3 arquitecturas A/B/C, 7 calidades JPEG, 4 condiciones de red con ancho de banda y latencia |

### Decisiones técnicas tomadas
- `config.py` expone **diccionarios indexados** `cheng2020_model_paths[q]` y `yolo_model_paths[variant]` para iterar programáticamente sobre calidades/variantes desde los notebooks.
- Dos funciones de utilidad con validación: `get_cheng2020_log_dir(q)` y `get_yolo_log_dir(variant)`.
- Los λ por calidad se basan en los valores de referencia de CompressAI (paper Cheng 2020) y son el punto de partida para Optuna en el notebook 4.1.
- Se eliminó el `print("Current working directory: ...")` que se ejecutaba en cada `import config`.
- Las condiciones de red se alinearon exactamente entre `config.yaml` y `NETWORK_CONDITIONS` en `throughput.py` (sesión siguiente).

### Pendientes
- Los valores de λ deben verificarse empíricamente con el dataset de residuos; el notebook 4.1 (Optuna) los ajustará.
- Las rutas de imágenes YOLO (`data/processed/yolo_*/`) requieren que los datasets estén preparados (notebook 0.0 y 2.0).

---

## Sesión 4 — Módulos de métricas y experimento de throughput

### Objetivo
Ampliar `src/utils/metrics.py` con `compute_bpp()` (ya movido) y un wrapper de mAP@0.5
sobre resultados YOLO. Crear desde cero `src/utils/throughput.py` con las funciones para
medir las tres arquitecturas del experimento central.

### Archivos modificados/creados
| Archivo | Acción |
|---|---|
| `src/utils/metrics.py` | Limpieza de bloques comentados heredados; orden de imports corregido; agregadas `compute_map50()` y `compute_map50_from_dict()` |
| `src/utils/throughput.py` | Creado desde cero (~270 líneas) |

### Contenido de `throughput.py`

| Símbolo | Descripción |
|---|---|
| `FrameResult` | Dataclass con todos los campos medidos por frame |
| `NetworkCondition` | Dataclass con bandwidth y latencia; método `transmission_time_ms(n_bytes)` |
| `NETWORK_CONDITIONS` | Dict con 4 condiciones: local (75 Mbps), starlink (50 Mbps), cellular_4g (15 Mbps), limited (2 Mbps) |
| `measure_arch_a()` | JPG q=100; psnr/ssim=NaN (no hay degradación de referencia) |
| `measure_arch_b()` | JPG q variable; mide PSNR/SSIM vs original |
| `measure_arch_c()` | Usa `model.compress()` / `model.decompress()` nativo de CompressAI (bytes reales de bitstream); `torch.cuda.synchronize()` para timing correcto en GPU |
| `run_experiment()` | Itera frames × arquitecturas × calidades |
| `results_to_dataframe()` | Convierte lista de `FrameResult` a pandas DataFrame |

### Decisiones técnicas tomadas
- **Arquitectura C usa el flujo nativo de CompressAI** (`compress()`/`decompress()`) en lugar del flujo manual con `g_a`/`entropy_bottleneck` que había en `inference.py` heredado. Esto produce bytes reales del bitstream (no tensores en memoria), por lo que `bytes_transmitted` refleja lo que realmente viajaría por la red.
- `measure_arch_a()` devuelve `psnr=NaN` y `ssim=NaN` porque el JPEG q=100 es el propio baseline de referencia; no tiene sentido compararlo consigo mismo.
- `compute_map50()` maneja dos variantes del objeto `Results` de Ultralytics (`.box.map50` y `.results_dict`) porque la API de Ultralytics varía entre versiones.
- El tiempo de transmisión simulado es determinista (no añade ruido aleatorio) para que los experimentos sean reproducibles.

### Pendientes
- `measure_arch_c()` asume que el modelo ya tiene `update_entropy_bottleneck()` llamado (requerimiento de CompressAI para inferencia). Documentar esto en el notebook 8.0.
- El notebook 8.0 debe decidir si se mide sobre el conjunto de test completo o sobre una muestra representativa por clase.

---

## Sesión 5 — Módulo YOLO completo

### Objetivo
Crear el módulo `src/models/yolo_model/` con los cuatro archivos: tres scripts de
entrenamiento para las variantes baseline/reconstructed/mixed e `inference.py`.

### Archivos creados
| Archivo | Descripción |
|---|---|
| `src/models/yolo_model/__init__.py` | Marcador de paquete |
| `src/models/yolo_model/train_baseline.py` | `train_yolo_baseline()` — entrenado sobre imágenes originales |
| `src/models/yolo_model/train_reconstructed.py` | `train_yolo_reconstructed()` — entrenado sobre imágenes reconstruidas por CompressAI |
| `src/models/yolo_model/train_mixed.py` | `train_yolo_mixed()` — entrenado sobre originales + reconstruidas |
| `src/models/yolo_model/inference.py` | `load_model()`, `predict()`, `validate()`, `predict_single()` |

### API pública de `inference.py`

| Función/Clase | Uso |
|---|---|
| `Detection` | Dataclass: class_id, class_name, confidence, bbox_xyxy, bbox_xywhn, timestamp_ms |
| `FrameDetections` | Dataclass: frame_index, inference_time_ms, lista de Detection |
| `CLASS_NAMES` | Lista de 7 clases leída de `config["data"]["classes"]` |
| `load_model(variant)` | Carga `best.pt` según rutas del config; `FileNotFoundError` descriptivo si no existe |
| `predict()` | Acepta PIL, tensor, path o lista; devuelve lista de `FrameDetections` |
| `validate()` | Corre `model.val()`; devuelve map50, map50_95, precision, recall, results_dict |
| `predict_single()` | Optimizada para throughput: timing con `perf_counter` + `cuda.synchronize()` |

### Decisiones técnicas tomadas
- Los tres scripts de entrenamiento son **thin wrappers sobre Ultralytics** que leen todos los hiperparámetros desde `config["yolo"]`. No reimplementan el loop de entrenamiento (Ultralytics lo gestiona internamente, incluyendo early stopping, checkpointing y TensorBoard).
- El parámetro `project`/`name` de Ultralytics se alinea con `models/trained/yolo_<variant>/train/weights/best.pt`, que coincide exactamente con las rutas en `config["models"]["yolo"]`.
- `predict_single()` existe como función separada de `predict()` porque en el experimento de throughput se necesita el tiempo de inferencia medido con `perf_counter` (más preciso para frames individuales que `result.speed["inference"]` de Ultralytics).
- `CLASS_NAMES` se lee del config en lugar de estar hardcodeada para que un cambio en las clases del dataset se propague automáticamente.
- Se acepta `override: dict` en los tres train scripts para facilitar ejecuciones de prueba desde notebook con `{'epochs': 5, 'batch': 4}`.

### Pendientes
- Los datasets YOLO en formato Ultralytics (`dataset.yaml` por variante) deben generarse en el notebook 2.0 / 6.0.
- Evaluar si conviene usar transfer learning desde YOLO-baseline como punto de partida para YOLO-reconstructed y YOLO-mixed (pasar `weights=ruta_a_best.pt`).
- El notebook 7.0/7.1/7.2 debe crearse con el contenido correspondiente.

---

## Sesión 6 — Dependencias y documentación de instalación

### Objetivo
Agregar `ultralytics` y `pycocotools` a `requirements.txt`, verificar el entorno
conda activo (`image-recon-dl-env`, Python 3.10.15) e instalar los paquetes faltantes.
Reescribir `docs/getting-started.rst` con instrucciones completas de reproducción.

### Archivos modificados/creados
| Archivo | Acción |
|---|---|
| `requirements.txt` | Agregados `ultralytics==8.4.30` y `pycocotools==2.0.11`; eliminado comentario de versión de albumentations |
| `docs/getting-started.rst` | Reescrito completamente (placeholder de 7 líneas → ~200 líneas) |
| `sessions/resumen_ejecutivo.md` | Creado (este archivo) |

### Estado del entorno tras la sesión
El entorno `image-recon-dl-env` tiene todas las dependencias del proyecto instaladas:

| Paquete | Versión instalada |
|---|---|
| Python | 3.10.15 |
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| compressai | 1.2.6 |
| ultralytics | 8.4.30 (instalado en esta sesión) |
| pycocotools | 2.0.11 (instalado en esta sesión) |
| optuna | 4.1.0 |
| tensorboard | 2.17.1 |
| torchmetrics | 1.6.0 |

### Contenido de `getting-started.rst`
Cubre: requisitos de hardware, creación del entorno conda con comando exacto de
PyTorch CUDA 12.1, verificación del entorno, tabla de 14 dependencias con roles,
estructura de datos esperada, secuencia completa de 20 notebooks (0.0–11.0),
comandos TensorBoard, diagrama ASCII del pipeline en producción, licencias y datos de contacto.

### Decisiones técnicas tomadas
- `pycocotools` se pinea a `2.0.11` (versión instalada) porque la API de COCO no es estable entre versiones menores.
- `ultralytics` se pinea a `8.4.30` por la misma razón — la API de `results.box.map50` puede cambiar.
- `albumentations` se dejó sin pinar (era `#==1.4.11` comentado) dado que el entorno tiene 1.4.21 y funciona correctamente; pinar a 1.4.11 sería regresivo.
- El entorno conda del proyecto es `image-recon-dl-env` (Python 3.10.15); ese nombre se documenta en `getting-started.rst`.

### Pendientes
- `test_environment.py` heredado del cookiecutter no verifica las dependencias específicas del proyecto (CompressAI, ultralytics, pycocotools). Sería conveniente actualizarlo.
- Documentar el proceso de instalación en el MacBook M4 Pro (dispositivo de borde) con soporte MPS en lugar de CUDA.

---

## Sesión 7 — Config de datasets externos y notebooks 0.0 a 1.3

### Objetivo
Agregar la sección `external_datasets` al config centralizado con los Google Drive IDs
de los tres datasets públicos y el placeholder del dataset del dron. Crear los primeros
cinco notebooks del proyecto (0.0 a 1.3) siguiendo la convención de nombres del CLAUDE.md,
reutilizando el código del proyecto de cultivos y adaptándolo al nuevo dominio.

### Archivos modificados/creados
| Archivo | Acción |
|---|---|
| `src/config.yaml` | Agregada sección `external_datasets` con gdrive_id, dest y format para los 4 datasets |
| `src/config.py` | Agregado shortcut `external_datasets = config["external_datasets"]` |
| `notebooks/0.0-jorge.ceferino.valdez-data-download-and-preparation.ipynb` | Creado |
| `notebooks/1.0-jorge.ceferino.valdez-image-analysis-and-format-conversion.ipynb` | Creado |
| `notebooks/1.1-jorge.ceferino.valdez-image-loading-and-visualization.ipynb` | Creado |
| `notebooks/1.2-jorge.ceferino.valdez-image-preprocessing-and-scaling.ipynb` | Creado |
| `notebooks/1.3-jorge.ceferino.valdez-dataset-split.ipynb` | Creado |
| `archive/notebooks/0.0-Jorge.Ceferino.Valdez-data-download-and-preparation.ipynb` | Movido desde `notebooks/` |
| `archive/notebooks/1.0-Jorge.Ceferino.Valdez-image-analysis_and_format_conversion.ipynb` | Movido desde `notebooks/` |
| `archive/notebooks/1.1-Jorge.Ceferino.Valdez-image-loading-and-initial-visualization.ipynb` | Movido desde `notebooks/` |
| `archive/notebooks/1.2-Jorge.Ceferino.Valdez-image-preprocessing-and-scaling.ipynb` | Movido desde `notebooks/` |
| `archive/notebooks/1.3-Jorge.Ceferino.Valdez-division.ipynb` | Movido desde `notebooks/` |

### Contenido de cada notebook

| Notebook | Reutiliza de cultivos | Adaptaciones clave |
|---|---|---|
| 0.0 | Lógica de descarga gdown + extracción zip | Itera sobre `config["external_datasets"]` en lugar de un solo ID; agrega `convert_coco` para COCO→YOLO; celda TODO para drone_captures condicionada a gdrive_id vacío |
| 1.0 | Loop de análisis de formatos + conversión JPG→PNG | Loop multi-dataset sobre `config["external_datasets"]`; histograma de resoluciones; grid de 8 imágenes por dataset; sección TODO drone condicionada a existencia en disco |
| 1.1 | Carga PIL/OpenCV + análisis de calidad (borrosas, sobreexpuestas) | Agrega verificación explícita de `pad_image_to_multiple()` con assert sobre múltiplo de 64; visualización before/after del padding |
| 1.2 | `redimensionar_imagen` + escalado con ThreadPoolExecutor | Lee `target_size` desde `config["compressai"]["training"]["image_size"]`; relleno negro centrado preservando aspect ratio; sección TODO drone |
| 1.3 | `train_test_split` estratificado + guardado CSV | Lee `random_seed`, `val_size`, `test_size` desde config; verifica no-fuga entre splits; verifica que todas las rutas del CSV existen en disco |

### Decisiones técnicas tomadas
- **`external_datasets` reemplaza `data.public_datasets`**: la sección anterior en `config.yaml` tenía metadatos (nombre, n_images) pero no los IDs de descarga. La nueva sección `external_datasets` es la fuente canónica para todo el pipeline de descarga; `data.public_datasets` se mantuvo como referencia documental pero no se usa programáticamente.
- **Celda TODO completamente escrita**: la lógica de descarga del dataset del dron está implementada en su totalidad; solo está gateada por `if not drone_gdrive_id`. En cuanto se complete el campo en `config.yaml`, la celda se ejecuta sin modificar código.
- **`convert_coco` de Ultralytics**: se eligió sobre una implementación propia porque es la misma herramienta que usan los pipelines YOLO y genera exactamente el formato esperado por `model.train()`.
- **Relleno negro centrado en 1.2**: la función `redimensionar_imagen` preserva el aspect ratio y rellena con negro. Alternativa rechazada: redimensionar directamente a 256×256 sin preservar ratio, que distorsionaría las proporciones de los residuos y podría afectar la detección.
- **Splits 70/15/15**: el config tiene `val_size=0.15` y `test_size=0.10` (herencia del proyecto de cultivos). En el notebook 1.3 se ajustó la lógica para que ambos splits sean iguales (15 % cada uno), dividiendo el bloque temp al 50/50. El config se dejó con los valores originales para no romper compatibilidad; el notebook es la fuente real de la partición.

### Estado de carpetas al cierre de la sesión

**`notebooks/`** — contiene los notebooks activos del nuevo proyecto (0.0–1.3) más los del proyecto de cultivos a partir de 2.0 que aún se usarán como referencia:
```
notebooks/
├── 0.0-jorge.ceferino.valdez-data-download-and-preparation.ipynb        ← nuevo
├── 1.0-jorge.ceferino.valdez-image-analysis-and-format-conversion.ipynb ← nuevo
├── 1.1-jorge.ceferino.valdez-image-loading-and-visualization.ipynb      ← nuevo
├── 1.2-jorge.ceferino.valdez-image-preprocessing-and-scaling.ipynb      ← nuevo
├── 1.3-jorge.ceferino.valdez-dataset-split.ipynb                        ← nuevo
├── 2.0-Jorge.Ceferino.Valdez-model-training-convolutional-autoencoder-model.ipynb  ← cultivos (referencia)
├── 3.0-Jorge.Ceferino.Valdez-hyperparameter-optimization.ipynb          ← cultivos (referencia)
├── 4.0-Jorge.Ceferino.Valdez-final-training-with-optimized-hyperparameters.ipynb   ← cultivos (referencia)
├── 5.0-Jorge.Ceferino.Valdez-inference-testing-with-final-model.ipynb   ← cultivos (referencia)
├── 5.1-jorge.Ceferino.Valdez-compute-resource.ipynb                     ← cultivos (referencia)
├── 6.0-Jorge.Ceferino.Valdez-compressai-installation-and-setup.ipynb    ← cultivos (referencia)
├── 7.0-Jorge.Ceferino.Valdez-compressai-model-training.ipynb            ← cultivos (referencia)
├── 7.1-Jorge.Ceferino.Valdez-compressai-model-optuna.ipynb              ← cultivos (referencia)
├── 8.1-Jorge.Ceferino.Valdez-compressai-save-compressed-image.ipynb     ← cultivos (referencia)
├── 9.0-Jorge.Ceferino.Valdez-compressai-reconstruct-compressed-image.ipynb ← cultivos (referencia)
├── 9.1-Jorge.Ceferino.Valdez-compressai-compute-resource.ipynb          ← cultivos (referencia)
├── 10.0-jorge.ceferino.valdez-baseline-tensorborad-log-to-csv.ipynb     ← cultivos (referencia)
├── 11.0-jorge.ceferino.valdez-compressai-tensorborad-log-to-csv.ipynb   ← cultivos (referencia)
├── 12.0-jorge.ceferino.valdez-comparative-graphs-of-the-models.ipynb    ← cultivos (referencia)
├── misc_plot_optuna_baseline.ipynb
├── misc_plot_optuna_baseline_2.ipynb
├── misc_plot_optuna_cheng2020.ipynb
└── prueba.ipynb
```

**`archive/notebooks/`** — equivalentes del proyecto de cultivos ya adaptados en esta sesión:
```
archive/notebooks/
├── 0.0-Jorge.Ceferino.Valdez-data-download-and-preparation.ipynb
├── 1.0-Jorge.Ceferino.Valdez-image-analysis_and_format_conversion.ipynb
├── 1.1-Jorge.Ceferino.Valdez-image-loading-and-initial-visualization.ipynb
├── 1.2-Jorge.Ceferino.Valdez-image-preprocessing-and-scaling.ipynb
└── 1.3-Jorge.Ceferino.Valdez-division.ipynb
```

### Pendientes
- **Ejecutar los notebooks** una vez que los datasets estén en Google Drive y los IDs sean accesibles. El notebook 0.0 fallará sin conectividad a Drive.
- **`data.public_datasets`** en config.yaml es ahora redundante con `external_datasets`; evaluar si se elimina en la próxima sesión.
- **`val_size` y `test_size` en config.yaml** no reflejan la partición real 70/15/15; corregir a `val_size: 0.15, test_size: 0.15` en la próxima sesión.
- A medida que se creen los notebooks 2.0–11.0, mover sus equivalentes del proyecto de cultivos a `archive/notebooks/`.

---

## Sesión 8 — Reemplazo de Beach-Litter-UAV por TACO_TN_UAV_2

### Objetivo
Reemplazar Beach-Litter-UAV en todo el proyecto por TACO_TN_UAV_2 (Roboflow, CC BY 4.0),
que tiene 12× más imágenes (6460 vs 534), splits ya preparados y clases más alineadas
con el dominio del proyecto. Actualizar config, notebook 0.0 y documentación.

### Archivos modificados
| Archivo | Acción |
|---|---|
| `src/config.yaml` | `external_datasets.beach_litter_uav` reemplazado por `taco_tn_uav_2` con gdrive_id, nc, names, class_mapping, license, source_url; agregados campos `annotations` e `images_dir` a uavvaste y taco; `data.public_datasets` actualizado |
| `src/config.py` | Sin cambios (ya expone `external_datasets` correctamente) |
| `notebooks/0.0-jorge.ceferino.valdez-data-download-and-preparation.ipynb` | Reescrito con 4 secciones separadas; lógica específica por dataset |
| `CLAUDE.md` | Sección bootstrap actualizada con TACO_TN_UAV_2, mapeo de clases y limitaciones |
| `docs/getting-started.rst` | Estructura de datos y licencias actualizadas |
| `references/README_COMPLETO.md` | Tabla de datasets y flujo de etiquetado actualizados; nota sobre Beach-Litter-UAV |

### Estructura del notebook 0.0 reescrito

| Sección | Dataset | Acción principal |
|---|---|---|
| 1 | TACO_TN_UAV_2 | Descarga → verifica estructura train/valid/test → aplica class_mapping → genera `data_remapped.yaml` |
| 2 | UAVVaste | Descarga → verifica `images/` (flat, 775 archivos) y `annotations/instances_default.json` → `convert_coco(labels_dir=annotations/)` |
| 3 | TACO | Descarga → verifica `batch_1/` a `batch_13/` y `annotations.json` en raíz → `convert_coco(labels_dir=dest_taco/)` |
| 4 | drone_captures | TODO condicionado a `gdrive_id` no vacío |

### Decisiones técnicas tomadas

| Decisión | Razón |
|---|---|
| TACO_TN_UAV_2 reemplaza Beach-Litter-UAV | 6460 imágenes vs 534; splits ya separados; CC BY 4.0; perspectiva aérea compatible |
| `aluminium wrap` → `film_bolsa` documentado como aproximación | Es papel aluminio, no bolsa plástica; introduce ruido en esa clase |
| `red_pesca` marcada como sin equivalente | No existe en ningún dataset público; clase exclusiva del dataset propio del dron |
| Campos `annotations` e `images_dir` en config para uavvaste/taco | Documenta la estructura real y evita hardcodear rutas en el notebook |
| Beach-Litter-UAV conservado en README como datos sin etiquetar | Puede ser útil para fine-tuning de dominio de CompressAI (ajuste visual de playa/costa) pero no para YOLO supervisado |
| `data_remapped.yaml` generado en notebook | Archivo de configuración YOLO listo para usar directamente en `model.train()` con las 7 clases del proyecto |

### Estructura real de los datasets (observada tras descarga)

| Dataset | Estructura | Archivos |
|---|---|---|
| UAVVaste | `images/` (flat), `annotations/instances_default.json` | 775 imágenes, 1 JSON |
| TACO | `batch_1/` a `batch_13/` (imágenes), `annotations.json` raíz | 1503 imágenes, 1 JSON |
| TACO_TN_UAV_2 | `train/`, `valid/`, `test/` (images+labels), `data.yaml` | 6460 imágenes (4527+654+1279) |

### Pendientes
- Ejecutar notebook 0.0 para descargar TACO_TN_UAV_2 y verificar la generación de `data_remapped.yaml`.
- Verificar que `convert_coco` de Ultralytics maneja correctamente `annotations.json` en la raíz del TACO (imágenes en subdirectorios `batch_*/`); ajustar ruta si hay error.
- Ejecutar notebook 0.0 completo con todos los datasets disponibles y documentar conteos reales.

---

## Estado global del proyecto al cierre de las 8 sesiones

### Módulos `src/` completos y listos para usar

```
src/
├── config.py                          ✓ reescrito + external_datasets shortcut
├── config.yaml                        ✓ reescrito + sección external_datasets
├── data/
│   └── make_dataset.py                ~ heredado, reemplazado funcionalmente por notebook 0.0
├── utils/
│   ├── datasets.py                    ✓ reutilizable sin cambios
│   ├── images.py                      ✓ pad_image_to_multiple() incorporada
│   ├── metrics.py                     ✓ limpio, compute_bpp + compute_map50
│   └── throughput.py                  ✓ creado desde cero
├── models/
│   ├── compressai_cheng2020_model/    ✓ renombrado, imports corregidos, compute_bpp centralizado
│   └── yolo_model/                    ✓ creado desde cero (4 archivos)
```

### Notebooks creados (numeración CLAUDE.md)

```
notebooks/
├── 0.0-jorge.ceferino.valdez-data-download-and-preparation.ipynb       ✓ (reescrito en sesión 8)
├── 1.0-jorge.ceferino.valdez-image-analysis-and-format-conversion.ipynb ✓
├── 1.1-jorge.ceferino.valdez-image-loading-and-visualization.ipynb      ✓
├── 1.2-jorge.ceferino.valdez-image-preprocessing-and-scaling.ipynb      ✓
├── 1.3-jorge.ceferino.valdez-dataset-split.ipynb                        ✓
└── (notebooks 2.0 a 11.0 pendientes)
```

### Pendientes globales para próximas sesiones

| Prioridad | Tarea |
|---|---|
| Alta | Ejecutar notebook 0.0 para descargar TACO_TN_UAV_2 y verificar `data_remapped.yaml` |
| Alta | Verificar que `convert_coco` maneja correctamente las rutas de TACO (images en `batch_*/`) |
| Alta | Completar `external_datasets.drone_captures.gdrive_id` en config.yaml cuando el dataset esté en Drive |
| Alta | Capturar dataset propio con DJI Mini 4 Pro + etiquetar con X-AnyLabeling (notebook 2.0) |
| Alta | Crear notebooks 2.0 a 11.0 |
| Alta | Eliminar notebooks del proyecto de cultivos (numeración antigua 0.0–12.0) |
| Media | Corregir `val_size`/`test_size` en config.yaml para reflejar la partición real 70/15/15 |
| Media | Eliminar `data.public_datasets` de config.yaml (redundante con `external_datasets`) |
| Media | Generar datasets YOLO en formato Ultralytics (dataset.yaml por variante) |
| Media | Actualizar `test_environment.py` para verificar CompressAI, ultralytics, pycocotools |
| Media | Documentar instalación en MacBook M4 Pro (MPS) en `getting-started.rst` |
| Baja | Considerar si `src/models/train_model.py` y `predict_model.py` (heredados, casi vacíos) se eliminan |
| Baja | `src/features/build_features.py` y `src/visualization/visualize.py` están vacíos |
