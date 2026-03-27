"""train_baseline.py — Entrenamiento de YOLO-baseline.

Variante: entrenado con imágenes originales, evaluado sobre imágenes originales.
Esta variante establece el techo de mAP alcanzable sin ningún tipo de compresión,
y sirve como referencia para la curva BPP vs mAP del experimento de throughput.

Uso desde notebook:
    from src.models.yolo_model.train_baseline import train_yolo_baseline
    metrics = train_yolo_baseline()
"""

from pathlib import Path

from ultralytics import YOLO

from src.config import config, models_dir, yolo_log_dirs


def train_yolo_baseline(
    data_yaml: str | Path | None = None,
    weights: str | None = None,
    override: dict | None = None,
) -> object:
    """Entrena YOLO-baseline sobre imágenes originales del dron.

    Lee todos los hiperparámetros desde src/config.yaml (sección `yolo`).
    El modelo resultante se guarda en models/trained/yolo_baseline/best.pt
    (gestionado por Ultralytics automáticamente en project/name).

    Args:
        data_yaml: Ruta al archivo .yaml del dataset en formato Ultralytics.
                   Si es None, se construye desde config['yolo']['variants']['baseline'].
        weights:   Pesos iniciales. Si es None usa la arquitectura base del config
                   (yolov8s.pt), lo que descarga los pesos COCO de Ultralytics.
        override:  Diccionario opcional para sobreescribir parámetros del config.
                   Ejemplo: {'epochs': 10, 'batch': 8} para una ejecución rápida.

    Returns:
        Objeto ultralytics.engine.results.Results con las métricas de la última
        época de validación.
    """
    yolo_cfg = config["yolo"]
    variant_cfg = yolo_cfg["variants"]["baseline"]

    if data_yaml is None:
        data_yaml = str(Path(variant_cfg["train_images"]).parent.parent / "dataset.yaml")

    if weights is None:
        weights = yolo_cfg["architecture"] + ".pt"

    log_dir = yolo_log_dirs["baseline"]()

    train_args = {
        "data":       data_yaml,
        "epochs":     yolo_cfg["epochs"],
        "imgsz":      yolo_cfg["imgsz"],
        "batch":      yolo_cfg["batch"],
        "patience":   yolo_cfg["patience"],
        "optimizer":  yolo_cfg["optimizer"],
        "lr0":        yolo_cfg["lr0"],
        "lrf":        yolo_cfg["lrf"],
        "workers":    yolo_cfg["workers"],
        "device":     yolo_cfg["device"],
        "seed":       yolo_cfg["random_seed"],
        "project":    str(models_dir().parent / "trained" / "yolo_baseline"),
        "name":       "train",
        "exist_ok":   True,
        "plots":      True,
        "save":       True,
    }

    if override:
        train_args.update(override)

    model = YOLO(weights)
    results = model.train(**train_args)

    print(f"[baseline] Entrenamiento completado.")
    print(f"[baseline] Pesos guardados en: {train_args['project']}/train/weights/best.pt")
    print(f"[baseline] mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', float('nan')):.4f}")

    return results
