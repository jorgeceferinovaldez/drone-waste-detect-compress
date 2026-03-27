# Reconstrucción de imágenes con aprendizaje profundo para identificar residuos
**Golfo San Jorge — Patagonia Argentina**  
Esp. Ing. Jorge Ceferino Valdez · Maestría en Informática y Sistemas · UNPA  
Directora: Dra. Norma Andrea Villagra · Codirector: Mg. Daniel Raúl Pandolfi

---

## Descripción general

Sistema de identificación de residuos costeros que integra tres componentes de aprendizaje profundo: compresión de imágenes (CompressAI), reconstrucción en servidor y detección de objetos (YOLO). La hipótesis central es que es posible transmitir únicamente la representación latente comprimida de una imagen —un volumen de bytes considerablemente menor que la imagen original— reconstruirla en un servidor con alta fidelidad y, sobre esa imagen reconstruida, ejecutar un detector de residuos con precisión comparable a la obtenida sobre la imagen original.

La comparación de throughput (bytes transmitidos, latencia, calidad de reconstrucción y precisión de detección) entre esta arquitectura y la transmisión de imagen cruda constituye el aporte experimental central de la tesis.

---

## Arquitectura del sistema

```
[Dron DJI Mini 4 Pro]
        |
        | video / frames
        v
[Dispositivo de borde]  ← MacBook M4 Pro 24 GB (banco de pruebas principal)
        |                  Raspberry Pi 5 / Jetson Orin Nano (alternativas embebidas)
        |  CompressAI encoder
        |  imagen → representación latente (bytes comprimidos)
        |
        | red (WiFi / Starlink Mini / canal limitado simulado)
        |  << comparación de throughput aquí >>
        v
[PC servidor]  ← Ryzen 9 5900X + RTX 4070 Ti Super 16 GB + 32 GB RAM
        |  CompressAI decoder
        |  bytes → imagen reconstruida
        |
        |  YOLO (entrenado con dataset propio de residuos costeros)
        v
[Detecciones]  →  clase · confianza · bounding box · timestamp
```

---

## Hardware del proyecto

| Rol | Dispositivo | Especificaciones clave |
|---|---|---|
| Captura de imágenes | DJI Mini 4 Pro | CMOS 1/1.3", 48 MP, video 4K@100 fps, control RC2 con pantalla |
| Captura de stream | Elgato Cam Link 4K | HDMI → USB, hasta 4K@30 fps |
| Dispositivo de borde | MacBook Pro M4 Pro | 14 núcleos CPU, Neural Engine 38 TOPS, 24 GB memoria unificada |
| Servidor / entrenamiento | PC Linux Pop OS 22.04 | Ryzen 9 5900X, RTX 4070 Ti Super 16 GB, 32 GB RAM DDR4, 2 TB NVMe |

### Nota sobre dispositivos de borde alternativos

La MacBook M4 Pro es el banco de pruebas principal para los experimentos de throughput. Para evaluar la viabilidad en edge computing embebido real (escenario más representativo de un sistema de monitoreo autónomo en campo), se pueden considerar las siguientes alternativas:

| Dispositivo | Acelerador | Tiempo est. CompressAI | Viabilidad |
|---|---|---|---|
| Raspberry Pi 3 | Ninguno | ~30-60 s/img | Inviable para tiempo real |
| Raspberry Pi 4 | Ninguno | ~8-15 s/img | Solo experimento offline |
| Raspberry Pi 5 | Ninguno | ~3-6 s/img | Aceptable para bajo fps |
| Orange Pi 5 | NPU 6 TOPS | ~0.3-1 s/img | Opción económica (~USD 90) |
| Jetson Orin Nano | CUDA 1024 núcleos | ~0.05-0.2 s/img | Recomendada para producción (~USD 250) |
| MacBook M4 Pro | Neural Engine 38 TOPS | ~0.03-0.1 s/img | Banco de pruebas principal |

> La Raspberry Pi 3 disponible no es viable como dispositivo de borde para este experimento. Su CPU ARM Cortex-A53 sin acelerador de inferencia tardaría entre 30 y 60 segundos por imagen al ejecutar el encoder de CompressAI, lo que sesgaría completamente la comparación de throughput. El cuello de botella sería el hardware, no el algoritmo.

---

## Software utilizado

| Biblioteca | Versión | Rol |
|---|---|---|
| Python | 3.10.12 (PC) | Entorno base |
| PyTorch | 2.3.1 | Framework de deep learning |
| CompressAI | 1.2.6.dev0 | Compresión y reconstrucción de imágenes con deep learning |
| Ultralytics | última estable | Entrenamiento e inferencia YOLO |
| TensorBoard | 2.17.1 | Monitoreo de entrenamiento |
| Optuna | 3.6.1 | Optimización de hiperparámetros |
| Scikit-learn | 1.5.1 | Métricas y evaluación |
| Matplotlib | 3.9.1 | Visualización de resultados |
| OpenCV | última estable | Procesamiento de video y frames |
| X-AnyLabeling | última estable | Etiquetado asistido por IA |

---

## Pipeline completo — etapas

```
ETAPA 1 · Captura de imágenes con dron
ETAPA 2 · Construcción del dataset y etiquetado
ETAPA 3 · Desarrollo del modelo CompressAI (compresión / reconstrucción)
ETAPA 4 · Entrenamiento del modelo YOLO sobre imágenes reconstruidas
ETAPA 5 · Experimento de comparación de throughput
ETAPA 6 · Validación del sistema integrado en campo
ETAPA 7 · Resultados y análisis
```

---

## Etapa 1 — Captura de imágenes con dron

**Hardware:** DJI Mini 4 Pro · sin modelo de IA involucrado

### Descripción

El DJI Mini 4 Pro vuela sobre la costa del Golfo San Jorge y captura imágenes y video de alta resolución. Esta etapa tiene dos objetivos simultáneos: obtener el material crudo para construir el dataset de entrenamiento (imágenes estáticas etiquetables), y capturar secuencias de video continuo para los experimentos de throughput en tiempo real. No hay ningún modelo de IA activo durante el vuelo.

### Configuración recomendada

Para maximizar la variabilidad del dataset y cubrir las condiciones reales de la costa patagónica:

- Altura de vuelo: 30-40 metros (perspectiva cenital, objetos distinguibles).
- Resolución de fotos para dataset: máxima (8064×6048 px), exportar a JPEG.
- Resolución de video para experimento de throughput: 4K@30 fps o 1080p@60 fps.
- Solapamiento entre fotogramas: 20-30% para cobertura continua sin puntos ciegos.
- Horarios variados: mañana, mediodía y tarde (el sol patagónico es muy lateral y genera sombras que dificultan la detección de objetos pequeños).

### Tipos de escenas a cubrir

- Zonas con residuos de distintos tipos: plástico, metal, vidrio, redes de pesca.
- Zonas sin residuos (fondo negativo): arena limpia, algas, rocas húmedas.
- Distintas condiciones: arena seca, arena húmeda, marea alta y baja.

La variedad de fondos negativos es tan importante como la de residuos: sin ella el modelo aprende a detectar cualquier objeto sobre arena, generando falsos positivos sobre algas y rocas.

### Entregable

- Carpeta `/raw_images/` con imágenes organizadas por fecha, zona y condición de luz.
- Carpeta `/raw_video/` con secuencias de video para experimentos de throughput.
- Log de vuelo: fecha, zona GPS aproximada, altura, condiciones climáticas.

---

## Etapa 2 — Construcción del dataset y etiquetado

**Hardware:** PC con RTX (o MacBook M4 Pro) · herramienta: X-AnyLabeling

### Descripción

Se construye el dataset de residuos costeros del Golfo San Jorge, que es un aporte original e independiente de la tesis. El dataset se usa en dos contextos distintos dentro del pipeline: como datos de dominio para el fine-tuning de CompressAI (imágenes sin etiquetar, para que el compresor aprenda la distribución visual de la costa patagónica), y como datos etiquetados para entrenar y evaluar YOLO.

El etiquetado se realiza con X-AnyLabeling, que permite auto-etiquetar con modelos preentrenados y corregir manualmente las predicciones.

### Bootstrap con datasets públicos

Para acelerar el auto-etiquetado inicial, se usan como semilla los siguientes datasets públicos:

| Dataset | Imágenes | Formato | Relevancia |
|---|---|---|---|
| Beach-Litter-UAV | 534 | YOLO nativo | Dron sobre playas, 5 clases, perspectiva cenital exacta |
| UAVVaste | 772 | COCO → convertir | Dron en entornos urbanos, residuos sobre suelo |
| TACO | ~5200 | COCO → convertir | General de residuos, incluye escenas de playa |

### Flujo de etiquetado con X-AnyLabeling

1. Cargar `best.pt` preentrenado con Beach-Litter-UAV como auto-labeler inicial.
2. Ejecutar batch auto-label sobre todas las imágenes propias con un clic.
3. Revisar cada imagen: aceptar predicciones correctas, ajustar las desplazadas, agregar residuos no detectados, eliminar falsos positivos.
4. Exportar en formato YOLO: carpetas `images/` y `labels/` más archivo `data.yaml`.

En la primera iteración se espera corregir el 40-60% de las predicciones, ya que el modelo no conoce las particularidades de la costa patagónica. Luego de entrenar el modelo propio con las primeras 500 imágenes, se puede reutilizar ese modelo como auto-labeler para las imágenes siguientes, reduciendo la corrección manual al 10-20%.

### Clases del dataset

| ID | Clase | Ejemplos |
|---|---|---|
| 0 | plastico_rigido | Botellas, bidones, envases duros |
| 1 | film_bolsa | Bolsas, nylon, film plástico |
| 2 | poliestireno | Telgopor, espuma expandida |
| 3 | metal | Latas, chatarra, cables |
| 4 | vidrio | Botellas rotas, fragmentos |
| 5 | red_pesca | Redes, sogas, cabos sintéticos |
| 6 | otros | Residuos no clasificables en las categorías anteriores |

### División del dataset

| Split | Porcentaje | Imágenes (~1000 total) |
|---|---|---|
| Entrenamiento | 70% | ~700 imágenes |
| Validación | 15% | ~150 imágenes |
| Test | 15% | ~150 imágenes |

### Entregable

- Carpeta `/dataset/` con estructura YOLO completa (`images/`, `labels/`, `data.yaml`).
- Subconjunto sin etiquetar para fine-tuning del dominio en CompressAI.
- Dataset publicado con DOI en Zenodo o IEEE DataPort (contribución citable independiente).

---

## Etapa 3 — Desarrollo del modelo CompressAI

**Hardware:** PC con RTX 4070 Ti Super · biblioteca: CompressAI 1.2.6

### Descripción

Esta es la etapa técnicamente más novedosa de la tesis. Se toma un modelo preentrenado de CompressAI y se realiza fine-tuning con las imágenes de residuos costeros propias. El objetivo es que el compresor aprenda a preservar especialmente las características visuales relevantes para la detección de residuos (bordes, texturas de plástico, colores de materiales artificiales) y no malgaste capacidad de codificación en el fondo (arena, agua, cielo).

### Estructura del autoencoder de CompressAI

El sistema de compresión tiene tres componentes secuenciales:

```
ENCODER  (corre en el dispositivo de borde — MacBook M4 Pro)
    imagen original  x  (H × W × 3)
        ↓  red convolucional / transformer de análisis
    representación latente  ŷ  (h × w × C, dimensión mucho menor)
        ↓  cuantización + codificación entrópica
    stream de bytes comprimidos  ←  esto es lo que viaja por la red

CANAL DE COMUNICACIÓN
    bytes comprimidos viajan por WiFi / Starlink Mini / red simulada

DECODER  (corre en el servidor — PC con RTX)
    stream de bytes comprimidos
        ↓  decodificación entrópica
    representación latente cuantizada  ŷ
        ↓  red convolucional / transformer de síntesis
    imagen reconstruida  x̂  ←  input para YOLO
```

### Modelos base disponibles en CompressAI

| Modelo | Arquitectura | Calidad (q) | Características |
|---|---|---|---|
| bmshj2018-factorized | Autoencoder + prior factorizado | 1-8 | Más rápido, menor calidad |
| mbt2018 | Autoencoder + prior hiperlatente | 1-8 | Balance calidad / velocidad |
| cheng2020-anchor | Autoencoder + módulos de atención | 1-6 | Mayor calidad, más lento |
| cheng2020-attn | Autoencoder + self-attention | 1-6 | Mejor para texturas complejas |

Se recomienda partir de `mbt2018` o `cheng2020-anchor` y evaluar cuál mantiene mejor la información relevante para YOLO tras la reconstrucción.

### Fine-tuning para el dominio costero

El fine-tuning se realiza con las imágenes sin etiquetar del dataset propio. La función de pérdida combina:

- **Pérdida de distorsión:** MSE o MS-SSIM entre imagen original y reconstruida.
- **Pérdida de tasa:** número de bits usados para codificar la representación latente.
- **Balance λ:** parámetro que controla el trade-off tasa/distorsión. Valores bajos de λ priorizan compresión agresiva; valores altos priorizan fidelidad.

### Métricas de evaluación de reconstrucción

| Métrica | Descripción | Objetivo |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio en dB | Mayor es mejor (>30 dB aceptable) |
| SSIM | Similitud estructural (0 a 1) | Mayor es mejor (>0.85 aceptable) |
| MS-SSIM | SSIM multi-escala | Más correlacionada con percepción humana |
| BPP | Bits por píxel transmitidos | Menor es mejor (eficiencia de compresión) |

### Curva rate-distortion

Para cada nivel de calidad q=1 a q=6 (o q=8 según el modelo), se mide el par (BPP, PSNR/SSIM). Esta curva rate-distortion sobre el dataset de residuos costeros es el resultado de referencia de esta etapa: permite elegir el punto de operación óptimo para los experimentos de throughput.

### Entregable

- Checkpoint PyTorch del modelo CompressAI fine-tuned por nivel de calidad.
- Curva rate-distortion sobre el dataset de residuos del Golfo San Jorge.
- Encoder exportado y verificado en la MacBook M4 Pro (dispositivo de borde).
- Decoder verificado en la PC con RTX (servidor).

---

## Etapa 4 — Entrenamiento del modelo YOLO sobre imágenes reconstruidas

**Hardware:** PC con RTX 4070 Ti Super · biblioteca: Ultralytics YOLOv8/v11

### Descripción

Se entrena el modelo YOLO específicamente sobre imágenes reconstruidas por CompressAI, no sobre imágenes originales. Esto es una decisión de diseño deliberada: el detector debe aprender la distribución de imágenes que efectivamente recibirá en producción, que son imágenes reconstruidas con posibles artefactos de compresión (borrosidad leve, pérdida de detalles finos), no imágenes perfectas que nunca llegarán al servidor.

### Estrategia de entrenamiento

1. Comprimir todo el set de entrenamiento con el encoder de CompressAI al nivel de calidad elegido.
2. Reconstruir con el decoder: se obtiene `images_reconstructed/`.
3. Usar esas imágenes reconstruidas junto con sus etiquetas YOLO originales para entrenar YOLOv8s o YOLOv11s.
4. Validar sobre imágenes reconstruidas del set de validación.
5. Evaluar sobre imágenes reconstruidas del set de test.

### Variantes del experimento comparativo YOLO

Se entrenan tres variantes para aislar el efecto de la compresión:

| Variante | Entrenado con | Inferencia sobre | Propósito |
|---|---|---|---|
| YOLO-baseline | Imágenes originales | Imágenes originales | Cota superior de rendimiento |
| YOLO-reconstructed | Imágenes reconstruidas | Imágenes reconstruidas | Sistema propuesto |
| YOLO-mixed | Originales + reconstruidas | Imágenes reconstruidas | Evaluación de robustez |

La diferencia en mAP entre YOLO-baseline y YOLO-reconstructed cuantifica el costo de detección introducido por la compresión. Si esa diferencia es pequeña (por ejemplo, menos de 2-3 puntos de mAP), la arquitectura de compresión es viable.

### Parámetros de entrenamiento sugeridos

| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo base | yolov8s.pt o yolov11s.pt | Balance velocidad / precisión para objetos pequeños |
| imgsz | 640 | Resolución estándar YOLO |
| epochs | 100 | Con early stopping (patience=20) |
| batch | 16 | Adecuado para 16 GB VRAM |
| conf umbral | 0.4 | Ajustar según precisión/recall deseado en campo |

### Métricas de evaluación

mAP@0.5, mAP@0.5:0.95, Precision, Recall y F1-score por clase. Se analiza especialmente qué clases de residuos se ven más afectadas por la compresión (por ejemplo, film plástico delgado puede perder bordes críticos a BPP bajos).

### Entregable

- `best.pt` de cada variante (YOLO-baseline, YOLO-reconstructed, YOLO-mixed).
- Tabla comparativa de métricas entre variantes por nivel de calidad q.
- Análisis por clase: qué tipos de residuos toleran mejor la compresión.

---

## Etapa 5 — Experimento de comparación de throughput

**Hardware:** MacBook M4 Pro (borde) + PC con RTX (servidor) + red

### Descripción

Este es el experimento de validación central de la tesis. Se mide y compara el rendimiento de dos arquitecturas de transmisión bajo las mismas condiciones de red: transmisión directa de imagen (baseline) versus transmisión de representación latente comprimida (sistema propuesto).

### Las dos arquitecturas comparadas

**Arquitectura A — Transmisión directa (baseline):**
```
MacBook M4 Pro → imagen JPEG → red → PC → YOLO → detecciones
```

**Arquitectura B — Transmisión comprimida (sistema propuesto):**
```
MacBook M4 Pro → CompressAI encoder → bytes latentes → red → PC → CompressAI decoder → YOLO → detecciones
```

### Variables medidas en cada experimento

| Variable | Unidad | Arquitectura A | Arquitectura B |
|---|---|---|---|
| Bytes transmitidos | KB / frame | Tamaño JPEG | Tamaño stream latente |
| Ratio de compresión | × | 1× (referencia) | Variable según q |
| Tiempo de encode | ms | — | Encoder en M4 Pro |
| Tiempo de transmisión | ms | Depende de red y tamaño | Depende de red y tamaño |
| Tiempo de decode | ms | — | Decoder en RTX |
| Latencia total pipeline | ms | tx | encode + tx + decode |
| Frames por segundo efectivos | fps | Limitado por tx | Limitado por encode + tx |
| PSNR imagen reconstruida | dB | — | Variable según q |
| mAP@0.5 detección | % | Baseline | Degradación por compresión |

### Niveles de calidad evaluados

El experimento se repite para q = 1, 2, 3, 4, 5, 6 de CompressAI. Esto genera una familia de curvas que muestran cómo evoluciona cada variable al aumentar la calidad de compresión. El resultado principal es la curva BPP vs mAP: el punto donde la compresión es suficientemente agresiva para reducir la carga de transmisión sin degradar la detección por debajo de un umbral aceptable.

### Condiciones de red evaluadas

| Escenario | Ancho de banda | Relevancia para la tesis |
|---|---|---|
| Red local (Ethernet / WiFi) | 50-100 Mbps | Laboratorio, condición ideal |
| 4G / LTE | 10-20 Mbps | Campo con cobertura celular |
| Starlink Mini | 20-100 Mbps variable | Costa patagónica remota |
| Canal limitado simulado | 1-5 Mbps | Escenario adverso, mayor beneficio de compresión |

La ventaja relativa de la Arquitectura B aumenta cuanto más estrecho es el canal: en red local la compresión puede no justificarse, pero en canal limitado o Starlink con variabilidad, la reducción de bytes transmitidos se traduce directamente en mayor fps efectivo.

### Entregable

- Tabla de resultados completa por q y por condición de red.
- Gráficas: BPP vs q, latencia vs q, mAP vs q, FPS efectivos vs ancho de banda.
- Curva rate-distortion extendida: BPP vs mAP (aporte central del paper).
- Análisis del punto óptimo de operación por escenario de red.

---

## Etapa 6 — Validación del sistema integrado en campo

**Hardware:** DJI Mini 4 Pro + MacBook M4 Pro + PC con RTX + red (Starlink Mini o WiFi)

### Descripción

Validación end-to-end del sistema completo en condiciones reales sobre la costa del Golfo San Jorge. Se ejecuta el pipeline completo durante vuelos reales: el dron captura video, la Elgato entrega el stream a la MacBook, CompressAI codifica cada frame, los bytes viajan por la red, el servidor reconstruye con el decoder y YOLO detecta los residuos. El resultado se visualiza en tiempo real con bounding boxes sobre la imagen reconstruida.

### Flujo de datos en campo

```
Dron DJI Mini 4 Pro
    ↓  señal de video HDMI
Control remoto RC2 con pantalla
    ↓  salida HDMI
Elgato Cam Link 4K
    ↓  USB-C
MacBook M4 Pro  →  CompressAI encoder (Neural Engine)
    ↓  bytes comprimidos
Red (WiFi local / Starlink Mini)
    ↓
PC con RTX  →  CompressAI decoder  →  YOLO  →  detecciones
    ↓
Video anotado en pantalla + CSV de detecciones guardado en disco
```

### Factores de robustez evaluados

- Iluminación: sol cenital del mediodía vs sol bajo patagónico de mañana y tarde.
- Oclusión: residuos parcialmente cubiertos por arena, algas o agua.
- Variabilidad de material: distintos colores, tamaños y estados de degradación.
- Variabilidad de red: latencia fluctuante de Starlink Mini en campo abierto.
- Altura de vuelo: comparar detección a 30 m vs 50 m de altitud.

### Qué se registra por sesión de vuelo

- Video original del dron (sin anotar) como respaldo.
- Video con bounding boxes superpuestos en tiempo real (anotado).
- CSV de detecciones: `timestamp`, `clase`, `confianza`, `x_centro`, `y_centro`, `ancho`, `alto` (coordenadas normalizadas).
- Log de red: bytes transmitidos por frame, latencia por frame, fps efectivos.
- Log de vuelo: altura, posición GPS aproximada, condiciones climáticas.

### Parámetro crítico: umbral de confianza

Valor inicial sugerido: `conf=0.4`. Si hay muchos falsos positivos sobre algas o sombras: subir a `0.5`. Si se pierden residuos reales pequeños: bajar a `0.35`. Ajustar en las primeras pasadas de campo antes de los vuelos definitivos.

### Entregable

- Video de vuelo real con detecciones superpuestas.
- CSV de detecciones con timestamps sincronizados.
- Log de throughput por sesión de vuelo.
- Informe de comparación entre métricas de laboratorio y campo real.

---

## Etapa 7 — Resultados y análisis

**Hardware:** PC o MacBook · herramientas: Python + Matplotlib + QGIS

### Descripción

Consolidación y análisis de todos los resultados experimentales. Se generan las figuras y tablas del documento de tesis, se compara el sistema con el estado del arte en compresión de imágenes con deep learning y en detección de residuos costeros, y se extraen las conclusiones sobre la viabilidad de la arquitectura compresión + reconstrucción + detección para monitoreo costero en entornos con recursos de transmisión limitados como la costa patagónica.

### Resultado principal: curva BPP vs mAP

La pregunta central que responde la tesis: ¿hasta qué nivel de compresión (BPP) se puede llevar una imagen de residuos costeros sin que la precisión del detector YOLO caiga por debajo de un umbral aceptable para el monitoreo ambiental? Esta curva es el aporte más significativo para la comunidad de visión por computadora aplicada a gestión de residuos y monitoreo costero.

### Análisis adicionales

- Impacto por clase: qué tipos de residuos toleran mejor la compresión (poliestireno blanco probablemente mejor que film plástico transparente).
- Análisis de artefactos: qué introduce la reconstrucción a distintos BPP y cómo afecta al detector.
- Comparación con JPEG: BPP vs mAP con compresión JPEG tradicional vs CompressAI.
- Viabilidad de edge computing: análisis de latencia por dispositivo de borde (M4 Pro, Jetson, RPi5).

### Mapas de distribución de residuos

Si el DJI Mini 4 Pro registra coordenadas GPS con timestamps sincronizables al CSV de detecciones: georreferenciar cada detección cruzando tiempo de detección con posición del dron en ese instante. Generar mapas de densidad de residuos sobre la costa del Golfo San Jorge en QGIS o Google Earth Engine. Comparar zonas de mayor acumulación entre distintas fechas de vuelo para detectar tendencias temporales.

### Aportes de la tesis

| Aporte | Descripción | Publicable en |
|---|---|---|
| Dataset etiquetado | Imágenes de residuos del Golfo San Jorge en formato YOLO | Zenodo, IEEE DataPort |
| Modelo CompressAI fine-tuned | Compresor/reconstructor ajustado a imágenes de residuos costeros patagónicos | GitHub + paper |
| Modelo YOLO para residuos costeros | Detector ajustado a perspectiva cenital de dron y condiciones locales | GitHub + paper |
| Curva BPP vs mAP | Cuánto se puede comprimir sin degradar la detección | Paper principal |
| Sistema de monitoreo integrado | Pipeline completo dron → compresión → reconstrucción → detección | Paper metodológico |

### Entregable final

- Documento de tesis completo.
- Repositorio GitHub con todo el código documentado y reproducible.
- Dataset publicado con DOI.
- Al menos un artículo enviado a conferencia o revista del área.

---

## Cronograma resumido (855 horas totales)

| Fase | Horas | Período estimado |
|---|---|---|
| Planificación | 60 h | Sep–Oct 2024 |
| Investigación | 115 h | Oct–Dic 2024 |
| Captura y preprocesamiento de imágenes | 110 h | Dic 2024 – Ene 2025 |
| Preparación del entorno | 110 h | Feb–Abr 2025 |
| Desarrollo del modelo de compresión y reconstrucción | 160 h | Abr–Jul 2025 |
| Desarrollo del modelo de detección YOLO | 135 h | Jul–Sep 2025 |
| Pruebas de validación | 105 h | Sep–Oct 2025 |
| Documentación y presentación final | 60 h | Oct–Dic 2025 |

---

## Estructura de carpetas del proyecto

```
drone-waste-detect-compress/
├── raw_images/              # Fotos originales del dron, sin procesar
├── raw_video/               # Video original para experimentos de throughput
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── dataset_reconstructed/   # Dataset comprimido y reconstruido por CompressAI
│   └── q{1..6}/             # Una carpeta por nivel de calidad
├── compressai_models/       # Checkpoints del modelo CompressAI fine-tuned
├── yolo_models/
│   ├── baseline/            # YOLO entrenado sobre imágenes originales
│   ├── reconstructed/       # YOLO entrenado sobre imágenes reconstruidas
│   └── mixed/               # YOLO entrenado con dataset mixto
├── experiments/
│   ├── rate_distortion/     # Curvas BPP vs PSNR/SSIM
│   ├── throughput/          # Resultados de comparación de throughput
│   └── field/               # Resultados de vuelos reales
├── results/
│   ├── figures/             # Gráficas para el documento de tesis
│   └── tables/              # Tablas de métricas
└── docs/
    └── propuesta_tesis.pdf
```

---

*Maestría en Informática y Sistemas · UNPA · Caleta Olivia, Santa Cruz, Argentina*
