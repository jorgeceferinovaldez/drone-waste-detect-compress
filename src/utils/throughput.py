"""throughput.py — Medición de throughput para el experimento central de la tesis.

Compara tres arquitecturas de transmisión imagen → detección YOLO:

  Arquitectura A: frame JPG calidad 100 → red → YOLO  (baseline puro)
  Arquitectura B: frame JPG calidades 10-95 → red → YOLO  (baseline compresión tradicional)
  Arquitectura C: cheng2020-anchor encoder → bytes → decoder → YOLO  (sistema propuesto)

La comparación clave es la curva BPP vs mAP@0.5 con las tres arquitecturas superpuestas.

Variables medidas por frame:
  bytes_transmitted   Bytes enviados por la red para este frame.
  encode_time_ms      Tiempo de codificación/compresión en el dispositivo de borde.
  decode_time_ms      Tiempo de decodificación/reconstrucción en el servidor.
  total_latency_ms    encode + transmisión simulada + decode.
  effective_fps       1000 / total_latency_ms.
  bpp                 Bits por píxel de los bytes transmitidos.
  psnr                PSNR entre original y reconstruida (NaN para arquitectura A).
  ssim                SSIM entre original y reconstruida (NaN para arquitectura A).
  map50               mAP@0.5 de YOLO sobre la imagen resultante (calculado a nivel de dataset).
"""

import io
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.utils.metrics import calculate_metrics, compute_bpp


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Resultado de procesar un único frame bajo una arquitectura dada."""
    architecture: str           # 'A', 'B' o 'C'
    quality: Optional[int]      # JPEG quality (A/B) o CompressAI q (C); None para A
    bytes_transmitted: int      # bytes enviados por la red
    encode_time_ms: float
    decode_time_ms: float
    total_latency_ms: float
    effective_fps: float
    bpp: float
    psnr: float                 # NaN si no hay reconstrucción (arquitectura A)
    ssim: float                 # NaN si no hay reconstrucción (arquitectura A)
    width: int
    height: int


@dataclass
class NetworkCondition:
    """Parámetros de una condición de red simulada."""
    label: str
    bandwidth_mbps: float
    latency_ms: float           # latencia de propagación (RTT/2)

    def transmission_time_ms(self, n_bytes: int) -> float:
        """Tiempo de transmisión en ms para n_bytes bajo este canal."""
        return (n_bytes * 8) / (self.bandwidth_mbps * 1_000_000) * 1000 + self.latency_ms


# Condiciones de red predefinidas (alineadas con config.yaml)
NETWORK_CONDITIONS = {
    "local": NetworkCondition("Local (WiFi)", bandwidth_mbps=75.0, latency_ms=1.0),
    "starlink": NetworkCondition("Starlink Mini", bandwidth_mbps=50.0, latency_ms=40.0),
    "cellular_4g": NetworkCondition("4G", bandwidth_mbps=15.0, latency_ms=30.0),
    "limited": NetworkCondition("Canal limitado", bandwidth_mbps=2.0, latency_ms=100.0),
}


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convierte tensor (C, H, W) en rango [0, 1] a PIL RGB."""
    return TF.to_pil_image(tensor.clamp(0, 1).cpu())


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convierte PIL RGB a tensor (C, H, W) en rango [0, 1]."""
    return TF.to_tensor(image)


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    """Codifica una imagen PIL como JPEG en memoria y devuelve los bytes."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_jpeg(jpeg_bytes: bytes) -> Image.Image:
    """Decodifica bytes JPEG y devuelve imagen PIL RGB."""
    return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")


def _num_pixels(tensor: torch.Tensor) -> int:
    """Número de píxeles de un tensor (C, H, W) o (B, C, H, W)."""
    if tensor.ndim == 4:
        return tensor.size(2) * tensor.size(3)
    return tensor.size(1) * tensor.size(2)


# ---------------------------------------------------------------------------
# Arquitectura A — JPG calidad 100 (baseline puro)
# ---------------------------------------------------------------------------

def measure_arch_a(
    frame_tensor: torch.Tensor,
    network: NetworkCondition,
    device: str = "cpu",
) -> FrameResult:
    """Mide throughput para arquitectura A: JPG q=100 → red → YOLO.

    No hay reconstrucción: la imagen que recibe YOLO es la descomprimida del JPEG.
    PSNR/SSIM se marcan como NaN porque la referencia de calidad es el propio JPEG.

    Args:
        frame_tensor: Tensor (C, H, W) en [0, 1], imagen del frame.
        network:      Condición de red a simular.
        device:       'cpu' o 'cuda'.

    Returns:
        FrameResult con los resultados medidos.
    """
    pil_image = _tensor_to_pil(frame_tensor)
    h, w = pil_image.height, pil_image.width

    # Encode
    t0 = time.perf_counter()
    jpeg_bytes = _encode_jpeg(pil_image, quality=100)
    encode_time_ms = (time.perf_counter() - t0) * 1000

    n_bytes = len(jpeg_bytes)

    # Decode
    t1 = time.perf_counter()
    _decode_jpeg(jpeg_bytes)
    decode_time_ms = (time.perf_counter() - t1) * 1000

    transmission_ms = network.transmission_time_ms(n_bytes)
    total_latency_ms = encode_time_ms + transmission_ms + decode_time_ms
    effective_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else float("inf")

    n_pixels = _num_pixels(frame_tensor)
    bpp = (n_bytes * 8) / n_pixels

    return FrameResult(
        architecture="A",
        quality=100,
        bytes_transmitted=n_bytes,
        encode_time_ms=encode_time_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        effective_fps=effective_fps,
        bpp=bpp,
        psnr=float("nan"),
        ssim=float("nan"),
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# Arquitectura B — JPG calidades variables (baseline compresión tradicional)
# ---------------------------------------------------------------------------

def measure_arch_b(
    frame_tensor: torch.Tensor,
    jpeg_quality: int,
    network: NetworkCondition,
    device: str = "cpu",
) -> FrameResult:
    """Mide throughput para arquitectura B: JPG q variable → red → YOLO.

    La imagen reconstruida (decodificada del JPEG) se compara con el original
    para calcular PSNR/SSIM.

    Args:
        frame_tensor:  Tensor (C, H, W) en [0, 1].
        jpeg_quality:  Calidad JPEG en [1, 95]. Valores típicos: 10, 20, 30, 50, 70, 85, 95.
        network:       Condición de red.
        device:        'cpu' o 'cuda'.

    Returns:
        FrameResult con los resultados medidos.
    """
    pil_image = _tensor_to_pil(frame_tensor)
    h, w = pil_image.height, pil_image.width

    # Encode
    t0 = time.perf_counter()
    jpeg_bytes = _encode_jpeg(pil_image, quality=jpeg_quality)
    encode_time_ms = (time.perf_counter() - t0) * 1000

    n_bytes = len(jpeg_bytes)

    # Decode
    t1 = time.perf_counter()
    reconstructed_pil = _decode_jpeg(jpeg_bytes)
    decode_time_ms = (time.perf_counter() - t1) * 1000

    transmission_ms = network.transmission_time_ms(n_bytes)
    total_latency_ms = encode_time_ms + transmission_ms + decode_time_ms
    effective_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else float("inf")

    n_pixels = _num_pixels(frame_tensor)
    bpp = (n_bytes * 8) / n_pixels

    reconstructed_tensor = _pil_to_tensor(reconstructed_pil).to(device)
    original_for_metric = frame_tensor.to(device)
    psnr_val, ssim_val = calculate_metrics(original_for_metric, reconstructed_tensor, device=device)

    return FrameResult(
        architecture="B",
        quality=jpeg_quality,
        bytes_transmitted=n_bytes,
        encode_time_ms=encode_time_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        effective_fps=effective_fps,
        bpp=bpp,
        psnr=psnr_val,
        ssim=ssim_val,
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# Arquitectura C — cheng2020-anchor (sistema propuesto)
# ---------------------------------------------------------------------------

def measure_arch_c(
    frame_tensor: torch.Tensor,
    compressai_model,
    network: NetworkCondition,
    device: str = "cpu",
) -> FrameResult:
    """Mide throughput para arquitectura C: cheng2020-anchor encoder → bytes → decoder → YOLO.

    Usa el flujo compress() / decompress() nativo de CompressAI, que produce
    bytes de bitstream real (no tensores en memoria).  El BPP se calcula sobre
    los bytes del bitstream transmitido.

    Args:
        frame_tensor:     Tensor (C, H, W) en [0, 1].
        compressai_model: Modelo cargado (cheng2020_anchor) en modo eval, en `device`.
        network:          Condición de red.
        device:           'cpu' o 'cuda'.

    Returns:
        FrameResult con los resultados medidos.
    """
    x = frame_tensor.unsqueeze(0).to(device)   # (1, C, H, W)
    h = x.size(2)
    w = x.size(3)
    n_pixels = h * w

    compressai_model.eval()

    # Encode — produce diccionario con 'strings' (bytes reales del bitstream)
    # y 'shape' necesario para el decoder
    with torch.no_grad():
        t0 = time.perf_counter()
        compressed = compressai_model.compress(x)
        if device != "cpu":
            torch.cuda.synchronize()
        encode_time_ms = (time.perf_counter() - t0) * 1000

    # Bytes transmitidos = suma de todos los strings del bitstream (y + z)
    n_bytes = sum(
        len(s)
        for stream_list in compressed["strings"]
        for s in stream_list
    )

    # Decode — reconstruye la imagen a partir del bitstream
    with torch.no_grad():
        t1 = time.perf_counter()
        decompressed = compressai_model.decompress(compressed["strings"], compressed["shape"])
        if device != "cpu":
            torch.cuda.synchronize()
        decode_time_ms = (time.perf_counter() - t1) * 1000

    transmission_ms = network.transmission_time_ms(n_bytes)
    total_latency_ms = encode_time_ms + transmission_ms + decode_time_ms
    effective_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else float("inf")

    bpp = (n_bytes * 8) / n_pixels

    x_hat = decompressed["x_hat"].clamp(0, 1)
    psnr_val, ssim_val = calculate_metrics(x.squeeze(0), x_hat.squeeze(0), device=device)

    return FrameResult(
        architecture="C",
        quality=None,           # se asigna externamente según el nivel q del modelo
        bytes_transmitted=n_bytes,
        encode_time_ms=encode_time_ms,
        decode_time_ms=decode_time_ms,
        total_latency_ms=total_latency_ms,
        effective_fps=effective_fps,
        bpp=bpp,
        psnr=psnr_val,
        ssim=ssim_val,
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# Ejecución de experimento completo sobre una lista de frames
# ---------------------------------------------------------------------------

def run_experiment(
    frames: list,
    compressai_models: dict,
    network: NetworkCondition,
    jpeg_qualities: list = None,
    compressai_qualities: list = None,
    device: str = "cpu",
) -> list:
    """Ejecuta el experimento de throughput completo sobre una lista de frames.

    Itera sobre todos los frames con arquitecturas A, B (por cada calidad JPEG)
    y C (por cada calidad CompressAI).

    Args:
        frames:               Lista de tensores (C, H, W) en [0, 1].
        compressai_models:    Diccionario {q: modelo} con los modelos cargados.
                              Ejemplo: {1: model_q1, 2: model_q2, ..., 6: model_q6}.
        network:              Condición de red a simular.
        jpeg_qualities:       Calidades JPEG para arquitecturas A y B.
                              Default: [100, 10, 20, 30, 50, 70, 85, 95].
        compressai_qualities: Niveles de calidad CompressAI a evaluar.
                              Default: [1, 2, 3, 4, 5, 6].
        device:               'cpu' o 'cuda'.

    Returns:
        Lista de FrameResult con todos los resultados.
    """
    if jpeg_qualities is None:
        jpeg_qualities = [100, 10, 20, 30, 50, 70, 85, 95]
    if compressai_qualities is None:
        compressai_qualities = [1, 2, 3, 4, 5, 6]

    results = []

    for i, frame in enumerate(frames):
        # Arquitectura A — JPG q=100
        results.append(measure_arch_a(frame, network=network, device=device))

        # Arquitectura B — JPG calidades variables
        for q in jpeg_qualities:
            if q == 100:
                continue  # ya cubierto por arquitectura A
            result = measure_arch_b(frame, jpeg_quality=q, network=network, device=device)
            results.append(result)

        # Arquitectura C — cheng2020-anchor por calidad
        for q in compressai_qualities:
            if q not in compressai_models:
                continue
            result = measure_arch_c(
                frame,
                compressai_model=compressai_models[q],
                network=network,
                device=device,
            )
            result.quality = q
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Conversión a DataFrame para análisis y guardado
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list):
    """Convierte una lista de FrameResult a un pandas DataFrame.

    Args:
        results: Lista de FrameResult.

    Returns:
        pandas.DataFrame con una fila por frame × arquitectura × calidad.
    """
    import pandas as pd
    from dataclasses import asdict
    return pd.DataFrame([asdict(r) for r in results])
