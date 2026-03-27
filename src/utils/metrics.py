import math

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# ---------------------------------------------------------------------------
# Métricas de reconstrucción de imagen
# ---------------------------------------------------------------------------

def calculate_metrics(original: torch.Tensor, decompressed: torch.Tensor, device: str = "cpu"):
    """Calcula PSNR y SSIM entre dos tensores de imagen.

    Args:
        original:     Tensor (C, H, W) o (B, C, H, W) en rango [0, 1].
        decompressed: Tensor de igual forma que original.
        device:       Dispositivo de cómputo ('cpu' o 'cuda').

    Returns:
        (psnr_value, ssim_value) como floats.
    """
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    if original.ndim == 3:
        original = original.unsqueeze(0)
    if decompressed.ndim == 3:
        decompressed = decompressed.unsqueeze(0)

    original = original.to(device)
    decompressed = decompressed.to(device)

    return psnr_metric(original, decompressed).item(), ssim_metric(original, decompressed).item()


# ---------------------------------------------------------------------------
# Métricas de compresión — CompressAI
# ---------------------------------------------------------------------------

def compute_bpp(out_net: dict) -> float:
    """Calcula bits por píxel (BPP) a partir de la salida del modelo CompressAI.

    BPP = -log2(likelihood) / num_píxeles, sumado sobre todos los canales latentes.
    Incluye tanto y_hat como z_hat (hiperprior).

    Args:
        out_net: Diccionario de salida de model.forward(), con claves
                 'x_hat' y 'likelihoods' ({'y': ..., 'z': ...}).

    Returns:
        BPP promedio del batch como float.
    """
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


def compute_compression_ratio(out_net: dict, original_image: torch.Tensor) -> float:
    """Ratio de compresión original/comprimido usando likelihoods de CompressAI.

    Args:
        out_net:        Salida de model.forward().
        original_image: Tensor de la imagen original (B, C, H, W).

    Returns:
        Ratio de compresión (> 1 significa que la imagen comprimida ocupa menos).
    """
    original_bits = original_image.numel() * original_image.element_size() * 8
    compressed_bits = sum(
        torch.log(likelihoods).sum() / -math.log(2)
        for likelihoods in out_net["likelihoods"].values()
    ).item()
    return original_bits / compressed_bits


def calculate_compression_ratio(original_size: int, latent_size: int) -> float:
    """Ratio de compresión simple a partir de tamaños en bytes."""
    return original_size / latent_size


def compute_compression_ratio_from_h5_data(original_image: torch.Tensor, compressed_data: dict) -> float:
    """Ratio de compresión calculado a partir de datos cargados desde HDF5.

    Args:
        original_image:  Tensor de la imagen original.
        compressed_data: Diccionario con claves y_hat, z_hat, y_hat_min/max,
                         y_shape, z_hat_min/max, z_shape.

    Returns:
        Ratio de compresión.
    """
    original_bits = original_image.numel() * original_image.element_size() * 8

    y_hat_bits = compressed_data["y_hat"].numel() * compressed_data["y_hat"].element_size() * 8
    z_hat_bits = compressed_data["z_hat"].numel() * compressed_data["z_hat"].element_size() * 8

    meta_bits = 0
    for key in ("y_hat_min", "y_hat_max", "z_hat_min", "z_hat_max"):
        val = compressed_data.get(key)
        if val is not None:
            meta_bits += val.element_size() * 8
    for key in ("y_shape", "z_shape"):
        val = compressed_data.get(key)
        if val is not None:
            meta_bits += int(np.prod(val)) * 8

    return original_bits / (y_hat_bits + z_hat_bits + meta_bits)


# ---------------------------------------------------------------------------
# mAP@0.5 — wrapper sobre resultados YOLO (Ultralytics)
# ---------------------------------------------------------------------------

def compute_map50(results) -> float:
    """Extrae mAP@0.5 de un objeto Results de Ultralytics YOLO.

    Funciona con la salida de model.val() o con una lista de objetos Results
    provenientes de model.predict().

    Args:
        results: Objeto ultralytics.engine.results.Results (de model.val())
                 o lista de Results (de model.predict()).

    Returns:
        mAP@0.5 como float en [0, 1]. Devuelve 0.0 si no hay detecciones
        o si results no contiene métricas de validación.

    Ejemplo (validación):
        metrics = yolo_model.val(data="dataset.yaml", ...)
        map50 = compute_map50(metrics)

    Ejemplo (predicción sobre lista de imágenes):
        # No aplica: predict() no calcula mAP directamente.
        # Usar model.val() para obtener mAP.
    """
    # Salida de model.val() — objeto con atributo .results_dict o .box.map50
    if hasattr(results, "box"):
        return float(results.box.map50)

    # Algunos wrappers exponen results_dict directamente
    if hasattr(results, "results_dict"):
        return float(results.results_dict.get("metrics/mAP50(B)", 0.0))

    # Lista de Results de model.predict() — no contiene mAP; devolver 0.0
    if isinstance(results, (list, tuple)):
        return 0.0

    return 0.0


def compute_map50_from_dict(results_dict: dict) -> float:
    """Extrae mAP@0.5 desde el diccionario results_dict de Ultralytics.

    Args:
        results_dict: Diccionario devuelto por metrics.results_dict, con clave
                      'metrics/mAP50(B)'.

    Returns:
        mAP@0.5 como float.
    """
    return float(results_dict.get("metrics/mAP50(B)", 0.0))
