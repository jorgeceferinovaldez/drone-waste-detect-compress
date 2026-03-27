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

## Estado global del proyecto al cierre de las 6 sesiones

### Módulos `src/` completos y listos para usar

```
src/
├── config.py                          ✓ reescrito, centralizado
├── config.yaml                        ✓ reescrito, 7 secciones
├── data/
│   └── make_dataset.py                ~ heredado, pendiente adaptar Google Drive ID
├── utils/
│   ├── datasets.py                    ✓ reutilizable sin cambios
│   ├── images.py                      ✓ pad_image_to_multiple() incorporada
│   ├── metrics.py                     ✓ limpio, compute_bpp + compute_map50
│   └── throughput.py                  ✓ creado desde cero
├── models/
│   ├── compressai_cheng2020_model/    ✓ renombrado, imports corregidos, compute_bpp centralizado
│   └── yolo_model/                    ✓ creado desde cero (4 archivos)
```

### Pendientes globales para próximas sesiones

| Prioridad | Tarea |
|---|---|
| Alta | Reemplazar notebooks con numeración 0.0–11.0 definida en CLAUDE.md |
| Alta | Actualizar `src/data/make_dataset.py` con Google Drive ID del dataset real y descarga de datasets públicos |
| Alta | Capturar dataset propio con DJI Mini 4 Pro + etiquetar con X-AnyLabeling |
| Media | Generar datasets YOLO en formato Ultralytics (dataset.yaml por variante) |
| Media | Actualizar `test_environment.py` para verificar CompressAI, ultralytics, pycocotools |
| Media | Documentar instalación en MacBook M4 Pro (MPS) en `getting-started.rst` |
| Baja | Considerar si `src/models/train_model.py` y `predict_model.py` (heredados, casi vacíos) se eliminan o se reescriben como dispatchers |
| Baja | `src/features/build_features.py` y `src/visualization/visualize.py` están vacíos |
