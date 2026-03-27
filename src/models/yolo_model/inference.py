"""inference.py — Inferencia con YOLO para detección de residuos costeros.

Provee funciones para:
  - Cargar cualquiera de las tres variantes de modelo (baseline/reconstructed/mixed).
  - Correr inferencia sobre una imagen PIL, tensor, path o lista de cualquiera de ellos.
  - Correr validación completa sobre un dataset para obtener mAP@0.5.
  - Extraer detecciones estructuradas (clase, confianza, bounding box, timestamp).

Las clases de residuos del dataset propio son:
    0: plastico_rigido
    1: film_bolsa
    2: poliestireno
    3: metal
    4: vidrio
    5: red_pesca
    6: otros
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from ultralytics import YOLO

from src.config import config, yolo_model_paths
from src.utils.metrics import compute_map50


# ---------------------------------------------------------------------------
# Estructura de resultado de detección
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Una detección individual de YOLO sobre un frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]      # [x1, y1, x2, y2] en píxeles
    bbox_xywhn: list[float]     # [cx, cy, w, h] normalizado [0, 1]
    timestamp_ms: float         # tiempo de inferencia desde inicio de secuencia


@dataclass
class FrameDetections:
    """Detecciones de un único frame."""
    frame_index: int
    inference_time_ms: float
    detections: list[Detection] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return len(self.detections)


# ---------------------------------------------------------------------------
# Nombres de clases del dataset propio
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = config["data"]["classes"]


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_model(variant: str, weights_path: str | Path | None = None) -> YOLO:
    """Carga un modelo YOLO entrenado por variante.

    Args:
        variant:      'baseline', 'reconstructed' o 'mixed'.
        weights_path: Ruta explícita al archivo .pt. Si es None, se usa la ruta
                      definida en config['models']['yolo'][variant].

    Returns:
        Modelo YOLO listo para inferencia (en modo eval).

    Raises:
        FileNotFoundError: Si el archivo de pesos no existe en la ruta indicada.
        ValueError:        Si variant no es uno de los valores válidos.
    """
    valid_variants = list(config["models"]["yolo"].keys())
    if variant not in valid_variants:
        raise ValueError(f"variant '{variant}' no válido. Opciones: {valid_variants}")

    if weights_path is None:
        weights_path = yolo_model_paths[variant]()
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pesos no encontrados en {weights_path}. "
            f"Entrenar primero con train_{variant}.py."
        )

    model = YOLO(str(weights_path))
    return model


# ---------------------------------------------------------------------------
# Inferencia sobre un frame o lista de frames
# ---------------------------------------------------------------------------

def predict(
    model: YOLO,
    source: Union[str, Path, Image.Image, torch.Tensor, list],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int | None = None,
    device: str | None = None,
    timestamp_offset_ms: float = 0.0,
) -> list[FrameDetections]:
    """Corre inferencia YOLO sobre una o varias imágenes.

    Args:
        model:              Modelo YOLO cargado con load_model().
        source:             Imagen(es) de entrada. Acepta: path str/Path, PIL Image,
                            tensor (C,H,W) o (B,C,H,W) en [0,1], lista de cualquiera.
        conf_threshold:     Umbral mínimo de confianza para incluir una detección.
        iou_threshold:      Umbral IoU para NMS.
        imgsz:              Tamaño de inferencia. None usa el del config (640).
        device:             'cpu', 'cuda', 'mps'. None usa el del config.
        timestamp_offset_ms: Offset de tiempo en ms para el primer frame,
                             útil cuando se procesa una secuencia de video.

    Returns:
        Lista de FrameDetections, una por imagen en source.
    """
    if imgsz is None:
        imgsz = config["yolo"]["imgsz"]
    if device is None:
        device = config["yolo"]["device"]

    # Normalizar source a lista para procesamiento uniforme
    if not isinstance(source, list):
        source = [source]

    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    frame_detections = []
    cumulative_ms = timestamp_offset_ms

    for frame_idx, result in enumerate(results):
        inference_time_ms = result.speed.get("inference", 0.0)
        cumulative_ms += inference_time_ms

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy  = result.boxes.xyxy.cpu().tolist()
            boxes_xywhn = result.boxes.xywhn.cpu().tolist()
            confs       = result.boxes.conf.cpu().tolist()
            cls_ids     = result.boxes.cls.cpu().int().tolist()

            for cls_id, conf, xyxy, xywhn in zip(cls_ids, confs, boxes_xyxy, boxes_xywhn):
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox_xyxy=[round(v, 2) for v in xyxy],
                    bbox_xywhn=[round(v, 6) for v in xywhn],
                    timestamp_ms=round(cumulative_ms, 2),
                ))

        frame_detections.append(FrameDetections(
            frame_index=frame_idx,
            inference_time_ms=round(inference_time_ms, 3),
            detections=detections,
        ))

    return frame_detections


# ---------------------------------------------------------------------------
# Validación sobre dataset completo — devuelve mAP@0.5
# ---------------------------------------------------------------------------

def validate(
    model: YOLO,
    data_yaml: str | Path,
    imgsz: int | None = None,
    batch: int | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    device: str | None = None,
) -> dict:
    """Valida el modelo sobre un dataset completo y devuelve métricas.

    Args:
        model:          Modelo YOLO cargado.
        data_yaml:      Ruta al dataset.yaml en formato Ultralytics.
        imgsz:          Tamaño de inferencia. None usa el del config.
        batch:          Batch size. None usa el del config.
        conf_threshold: Umbral de confianza para validación (default bajo para mAP).
        iou_threshold:  Umbral IoU para NMS durante validación.
        device:         Dispositivo. None usa el del config.

    Returns:
        Diccionario con:
            map50       mAP@0.5
            map50_95    mAP@0.5:0.95
            precision   Precisión media
            recall      Recall medio
            results_dict  Diccionario completo de Ultralytics
    """
    if imgsz is None:
        imgsz = config["yolo"]["imgsz"]
    if batch is None:
        batch = config["yolo"]["batch"]
    if device is None:
        device = config["yolo"]["device"]

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )

    return {
        "map50":       compute_map50(metrics),
        "map50_95":    float(metrics.box.map),
        "precision":   float(metrics.box.mp),
        "recall":      float(metrics.box.mr),
        "results_dict": metrics.results_dict,
    }


# ---------------------------------------------------------------------------
# Inferencia sobre stream de frames — para experimento de throughput
# ---------------------------------------------------------------------------

def predict_single(
    model: YOLO,
    frame: Union[Image.Image, torch.Tensor],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int | None = None,
    device: str | None = None,
) -> tuple[list[Detection], float]:
    """Infiere sobre un único frame y mide el tiempo de inferencia.

    Versión optimizada para el experimento de throughput: evita overhead de
    construir FrameDetections y devuelve directamente las detecciones y el
    tiempo medido con perf_counter (más preciso que result.speed para frames
    individuales).

    Args:
        model:          Modelo YOLO cargado.
        frame:          PIL Image o tensor (C,H,W) en [0,1].
        conf_threshold: Umbral de confianza.
        iou_threshold:  Umbral IoU.
        imgsz:          Tamaño de inferencia.
        device:         Dispositivo.

    Returns:
        (detections, inference_time_ms)
        detections: Lista de Detection (puede estar vacía).
        inference_time_ms: Tiempo de inferencia medido con perf_counter.
    """
    if imgsz is None:
        imgsz = config["yolo"]["imgsz"]
    if device is None:
        device = config["yolo"]["device"]

    t0 = time.perf_counter()
    results = model.predict(
        source=[frame],
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy  = result.boxes.xyxy.cpu().tolist()
        boxes_xywhn = result.boxes.xywhn.cpu().tolist()
        confs       = result.boxes.conf.cpu().tolist()
        cls_ids     = result.boxes.cls.cpu().int().tolist()

        for cls_id, conf, xyxy, xywhn in zip(cls_ids, confs, boxes_xyxy, boxes_xywhn):
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
                bbox_xyxy=[round(v, 2) for v in xyxy],
                bbox_xywhn=[round(v, 6) for v in xywhn],
                timestamp_ms=round(inference_time_ms, 2),
            ))

    return detections, inference_time_ms
