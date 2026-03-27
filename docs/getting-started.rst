Getting started
===============

Esta guía describe cómo reproducir el entorno del proyecto
**drone-waste-detect-compress** desde cero en un sistema Linux con GPU NVIDIA.

.. contents:: Contenidos
   :local:
   :depth: 2


Requisitos de hardware
----------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Dispositivo
     - Especificación mínima
   * - Servidor / entrenamiento
     - GPU NVIDIA con ≥ 12 GB VRAM (probado con RTX 4070 Ti Super 16 GB)
   * - Dispositivo de borde
     - MacBook Pro M4 Pro 24 GB o equivalente (solo inferencia CompressAI)
   * - RAM
     - ≥ 16 GB (32 GB recomendado)
   * - Almacenamiento
     - ≥ 20 GB libres para datos y modelos

.. note::
   La Raspberry Pi 3 (ARM Cortex-A53) **no es viable** para inferencia CompressAI:
   el encoder tarda 30–60 s/imagen sin acelerador de hardware.


Requisitos de software
----------------------

- Sistema operativo: Ubuntu 22.04 / Pop!_OS 22.04 (probado) o cualquier Linux x86-64
- CUDA: 12.1 o superior
- Miniconda / Anaconda (recomendado) o virtualenv
- Git


Creación del entorno conda
--------------------------

.. code-block:: bash

   # Clonar el repositorio
   git clone <url-del-repositorio> drone-waste-detect-compress
   cd drone-waste-detect-compress

   # Crear entorno con Python 3.10.15
   conda create -n image-recon-dl-env python=3.10.15 -y
   conda activate image-recon-dl-env

   # Instalar PyTorch 2.3.1 con soporte CUDA 12.1
   # (verificar versión exacta en https://pytorch.org/get-started/previous-versions/)
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
       --index-url https://download.pytorch.org/whl/cu121

   # Instalar el resto de dependencias
   pip install -r requirements.txt

.. note::
   CompressAI 1.2.6 requiere PyTorch ≥ 2.0. No instalar CompressAI antes de PyTorch
   porque ``pip install -r requirements.txt`` lo instala en orden correcto dado
   que ``torch`` aparece primero en el archivo.


Verificación del entorno
------------------------

.. code-block:: bash

   python test_environment.py

La salida esperada confirma Python 3.10, CUDA disponible y todas las dependencias
correctamente instaladas.

Para una verificación rápida manual:

.. code-block:: python

   import torch
   import compressai
   from ultralytics import YOLO
   import pycocotools

   print(torch.__version__)          # 2.3.1
   print(torch.cuda.is_available())  # True
   print(compressai.__version__)     # 1.2.6


Dependencias principales
------------------------

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Paquete
     - Versión
     - Uso
   * - ``torch``
     - 2.3.1
     - Backend de deep learning
   * - ``torchvision``
     - 0.18.1
     - Transformaciones de imagen
   * - ``compressai``
     - 1.2.6
     - Modelo cheng2020-anchor (compresión)
   * - ``ultralytics``
     - 8.4.30
     - YOLOv8s (detección de residuos)
   * - ``pycocotools``
     - 2.0.11
     - Conversión datasets COCO (UAVVaste, TACO)
   * - ``optuna``
     - ≥ 3.6
     - Optimización de hiperparámetros
   * - ``tensorboard``
     - 2.17.1
     - Visualización de métricas de entrenamiento
   * - ``albumentations``
     - ≥ 1.4
     - Aumentación de datos
   * - ``h5py``
     - 3.12.1
     - Almacenamiento de representaciones latentes
   * - ``opencv-contrib-python``
     - 4.10.0.84
     - Procesamiento de video (extracción de frames)
   * - ``torchmetrics``
     - ≥ 1.0
     - PSNR y SSIM
   * - ``pandas``
     - 2.2.2
     - Splits del dataset y resultados de throughput
   * - ``gdown``
     - 5.2.0
     - Descarga de datasets desde Google Drive


Estructura de datos
-------------------

Antes de ejecutar los notebooks es necesario tener los datos en la estructura
correcta. El notebook ``0.0`` descarga y organiza los datos automáticamente.
La estructura esperada es::

   data/
   ├── raw/
   │   └── drone_captures/data/    ← frames originales del dron DJI Mini 4 Pro
   ├── external/                   ← datasets públicos de bootstrap
   │   ├── Beach-Litter-UAV/       (534 imgs, formato YOLO nativo)
   │   ├── UAVVaste/               (772 imgs, formato COCO → convertir)
   │   └── TACO/                   (5200 imgs, formato COCO → convertir)
   ├── interim/
   │   └── dataset/                ← imágenes organizadas por clase
   │       ├── plastico_rigido/
   │       ├── film_bolsa/
   │       ├── poliestireno/
   │       ├── metal/
   │       ├── vidrio/
   │       ├── red_pesca/
   │       └── otros/
   └── processed/
       ├── train.csv
       ├── val.csv
       └── test.csv

Los datasets UAVVaste y TACO están en formato COCO y deben convertirse a
formato YOLO con ``pycocotools`` antes de usarlos (ver notebook ``0.0``).


Secuencia de notebooks
----------------------

Los notebooks están numerados en orden de ejecución:

.. list-table::
   :widths: 10 90
   :header-rows: 0

   * - ``0.0``
     - Descarga y preparación de datos (datasets públicos + capturas del dron)
   * - ``1.0``
     - Análisis de imágenes y conversión de formato
   * - ``1.1``
     - Carga y visualización inicial
   * - ``1.2``
     - Preprocesamiento y escalado a 256×256
   * - ``1.3``
     - División en splits (train/val/test)
   * - ``2.0``
     - Etiquetado con X-AnyLabeling (auto-label por lotes + corrección manual)
   * - ``3.0``
     - Setup y verificación de CompressAI cheng2020-anchor
   * - ``4.0``
     - Entrenamiento CompressAI (q=1 a q=6)
   * - ``4.1``
     - Optimización de hiperparámetros CompressAI con Optuna
   * - ``5.0``
     - Guardado de representaciones comprimidas a disco
   * - ``6.0``
     - Reconstrucción de imágenes desde representaciones comprimidas
   * - ``6.1``
     - Medición de recursos computacionales (CPU/GPU/RAM) de CompressAI
   * - ``7.0``
     - Entrenamiento YOLO-baseline (imágenes originales)
   * - ``7.1``
     - Entrenamiento YOLO-reconstructed (imágenes reconstruidas)
   * - ``7.2``
     - Entrenamiento YOLO-mixed (originales + reconstruidas)
   * - ``8.0``
     - Experimento de throughput (BPP vs mAP, tres arquitecturas)
   * - ``9.0``
     - Inferencia en campo con stream Elgato Cam Link 4K
   * - ``10.0``
     - Exportación de logs TensorBoard a CSV
   * - ``11.0``
     - Gráficas comparativas finales (curva BPP vs mAP)


TensorBoard
-----------

Los logs de entrenamiento se escriben en ``reports/logs/``, separados por
modelo y nivel de calidad:

.. code-block:: bash

   # Ver logs de cheng2020-anchor calidad 3
   tensorboard --logdir reports/logs/cheng2020_anchor_q3

   # Ver todos los modelos CompressAI comparados
   tensorboard --logdir reports/logs/ \
       --samples_per_plugin=images:100

   # Ver variantes YOLO
   tensorboard --logdir reports/logs/yolo_baseline


Pipeline completo en producción
--------------------------------

Una vez entrenados todos los modelos, el pipeline de inferencia en campo es::

   [Dron DJI Mini 4 Pro]
       ↓ video HDMI → Elgato Cam Link 4K
   [MacBook M4 Pro — dispositivo de borde]
       ↓ extracción de frames (1–5 fps)
       ↓ cheng2020-anchor encoder → bytes del bitstream
   [Red — WiFi / Starlink Mini]
       ↓
   [PC servidor — RTX 4070 Ti Super]
       ↓ cheng2020-anchor decoder → imagen reconstruida
       ↓ YOLOv8s
   [Detecciones] → clase · confianza · bounding box · timestamp

Ver notebook ``9.0`` para el código completo del stream en tiempo real.


Licencias
---------

- **Código**: MIT License — Copyright (c) 2024 Jorge Ceferino Valdez
- **Dataset propio**: CC BY 4.0 — publicado en Zenodo / IEEE DataPort
- **Datasets de bootstrap**:

  - Beach-Litter-UAV: ver licencia original del dataset
  - UAVVaste: ver licencia original del dataset
  - TACO: MIT License


Contacto
--------

| **Jorge Ceferino Valdez**
| Maestría en Informática y Sistemas — UNPA
| Caleta Olivia, Santa Cruz, Patagonia Argentina
| Directora: Dra. Norma Andrea Villagra (UNPA)
| Codirector: Mg. Daniel Raúl Pandolfi (UNPA)
