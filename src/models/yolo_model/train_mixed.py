"""train_mixed.py — Entrenamiento de YOLO-mixed.

Variante: entrenado con imágenes originales + reconstruidas (mezcla),
evaluado sobre imágenes reconstruidas.

Esta variante busca el mejor mAP posible sobre imágenes reconstruidas exponiéndole
al modelo tanto imágenes limpias como imágenes con artefactos de compresión durante
el entrenamiento. Se espera que sea igual o mejor que YOLO-reconstructed, y que
la brecha con YOLO-baseline cuantifique la degradación neta introducida por CompressAI.

Uso desde notebook:
    from src.models.yolo_model.train_mixed import train_yolo_mixed
    metrics = train_yolo_mixed()
"""

from pathlib import Path

from ultralytics import YOLO

from src.config import config, models_dir, yolo_log_dirs


def train_yolo_mixed(
    data_yaml: str | Path | None = None,
    weights: str | None = None,
    override: dict | None = None,
) -> object:
    """Entrena YOLO-mixed sobre imágenes originales + reconstruidas.

    Lee todos los hiperparámetros desde src/config.yaml (sección `yolo`).
    El modelo resultante se guarda en models/trained/yolo_mixed/best.pt.

    El dataset mixto debe estar preparado previamente combinando las imágenes
    de data/processed/yolo_baseline/ y data/processed/yolo_reconstructed/ en
    data/processed/yolo_mixed/ con un único dataset.yaml que apunte a ambas.

    Args:
        data_yaml: Ruta al archivo .yaml del dataset en formato Ultralytics.
                   Si es None, se construye desde config['yolo']['variants']['mixed'].
        weights:   Pesos iniciales. Si es None usa la arquitectura base del config
                   (yolov8s.pt). Para transfer learning desde YOLO-baseline, pasar
                   la ruta a models/trained/yolo_baseline/train/weights/best.pt.
        override:  Diccionario opcional para sobreescribir parámetros del config.

    Returns:
        Objeto ultralytics.engine.results.Results con las métricas de validación.
    """
    yolo_cfg = config["yolo"]
    variant_cfg = yolo_cfg["variants"]["mixed"]

    if data_yaml is None:
        data_yaml = str(Path(variant_cfg["train_images"]).parent.parent / "dataset.yaml")

    if weights is None:
        weights = yolo_cfg["architecture"] + ".pt"

    log_dir = yolo_log_dirs["mixed"]()

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
        "project":    str(models_dir().parent / "trained" / "yolo_mixed"),
        "name":       "train",
        "exist_ok":   True,
        "plots":      True,
        "save":       True,
    }

    if override:
        train_args.update(override)

    model = YOLO(weights)
    results = model.train(**train_args)

    print(f"[mixed] Entrenamiento completado.")
    print(f"[mixed] Pesos guardados en: {train_args['project']}/train/weights/best.pt")
    print(f"[mixed] mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', float('nan')):.4f}")

    return results
