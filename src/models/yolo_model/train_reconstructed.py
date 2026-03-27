"""train_reconstructed.py — Entrenamiento de YOLO-reconstructed.

Variante: entrenado con imágenes reconstruidas por cheng2020-anchor,
evaluado sobre imágenes reconstruidas.

Esta variante mide el mAP alcanzable cuando el modelo YOLO se adapta
específicamente a los artefactos de compresión de CompressAI, sin nunca
haber visto imágenes originales durante el entrenamiento.

Uso desde notebook:
    from src.models.yolo_model.train_reconstructed import train_yolo_reconstructed
    metrics = train_yolo_reconstructed()
"""

from pathlib import Path

from ultralytics import YOLO

from src.config import config, models_dir, yolo_log_dirs


def train_yolo_reconstructed(
    data_yaml: str | Path | None = None,
    weights: str | None = None,
    override: dict | None = None,
) -> object:
    """Entrena YOLO-reconstructed sobre imágenes reconstruidas por CompressAI.

    Lee todos los hiperparámetros desde src/config.yaml (sección `yolo`).
    El modelo resultante se guarda en models/trained/yolo_reconstructed/best.pt.

    Las imágenes reconstruidas deben estar generadas previamente con el notebook
    6.0 (compressai-reconstruct) para cada nivel de calidad q que se quiera evaluar.
    Se recomienda generar con el q que produzca el mejor balance BPP/mAP según
    la curva del experimento de throughput.

    Args:
        data_yaml: Ruta al archivo .yaml del dataset en formato Ultralytics.
                   Si es None, se construye desde config['yolo']['variants']['reconstructed'].
        weights:   Pesos iniciales. Si es None usa la arquitectura base del config
                   (yolov8s.pt). Para transfer learning desde YOLO-baseline, pasar
                   la ruta a models/trained/yolo_baseline/train/weights/best.pt.
        override:  Diccionario opcional para sobreescribir parámetros del config.

    Returns:
        Objeto ultralytics.engine.results.Results con las métricas de validación.
    """
    yolo_cfg = config["yolo"]
    variant_cfg = yolo_cfg["variants"]["reconstructed"]

    if data_yaml is None:
        data_yaml = str(Path(variant_cfg["train_images"]).parent.parent / "dataset.yaml")

    if weights is None:
        weights = yolo_cfg["architecture"] + ".pt"

    log_dir = yolo_log_dirs["reconstructed"]()

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
        "project":    str(models_dir().parent / "trained" / "yolo_reconstructed"),
        "name":       "train",
        "exist_ok":   True,
        "plots":      True,
        "save":       True,
    }

    if override:
        train_args.update(override)

    model = YOLO(weights)
    results = model.train(**train_args)

    print(f"[reconstructed] Entrenamiento completado.")
    print(f"[reconstructed] Pesos guardados en: {train_args['project']}/train/weights/best.pt")
    print(f"[reconstructed] mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', float('nan')):.4f}")

    return results
