import torch
import math
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# Calculate PSNR and SSIM
# def calculate_metrics(original, decompressed):
#     if original.max() > 1:
#         original = original / 255.0

#     if decompressed.max() > 1:
#         decompressed = decompressed / 255.0

#     psnr_value = psnr(original, decompressed)

#     win_size = min(original.shape[0], original.shape[1], 7)
#     win_size = win_size if win_size % 2 == 1 else win_size - 1

#     #ssim_value, _ = ssim(original, decompressed, win_size=win_size, full=True, data_range=1.0, channel_axis=-1)
#     ssim_value, _ = ssim(original, decompressed, win_size=win_size, full=True, data_range=1.0, multichannel=True)
#     return psnr_value, ssim_value

# Calculate PSNR and SSIM
# def calculate_metrics(original, decompressed):
#     if original.max() > 1:
#         original = original / 255.0

#     if decompressed.max() > 1:
#         decompressed = decompressed / 255.0

#     psnr_value = psnr(original, decompressed)

#     # Ensure win_size does not exceed image dimensions
#     min_dimension = min(original.shape[:2])
#     win_size = min(min_dimension, 7)
#     win_size = win_size if win_size % 2 == 1 else win_size - 1

#     if win_size < 1:  # Ensure win_size is at least 1
#         win_size = 1

#     #ssim_value, _ = ssim(original, decompressed, win_size=win_size, full=True, data_range=1.0, multichannel=True)
#     ssim_value, _ = ssim(original, decompressed, win_size=win_size, full=True, data_range=1.0, channel_axis=-1)
#     return psnr_value, ssim_value

# Calcular PSNR y SSIM
def calculate_metrics(original, decompressed, device="cpu"):
    # Inicializar las métricas en el dispositivo especificado
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    # Añadir dimensión de lote si es necesario
    if original.ndim == 3:
        original = original.unsqueeze(0)
    if decompressed.ndim == 3:
        decompressed = decompressed.unsqueeze(0)

    # Asegurarse de que los tensores estén en el dispositivo correcto
    original = original.to(device)
    decompressed = decompressed.to(device)

    # Calcular PSNR y SSIM
    psnr_value = psnr_metric(original, decompressed).item()
    ssim_value = ssim_metric(original, decompressed).item()

    return psnr_value, ssim_value

# Función para calcular la razón de compresión
def calculate_compression_ratio(original_size, latent_size):
    return original_size / latent_size

# Función para calcular el ratio de compresión usando las 'likelihoods'
def compute_compression_ratio(out_net, original_image):
    # Calcular el tamaño de la imagen original en bits
    original_size = original_image.numel() * original_image.element_size() * 8
    
    # Calcular el tamaño de la imagen comprimida en bits utilizando las "likelihoods"
    compressed_bits = sum(
        torch.log(likelihoods).sum() / -math.log(2)
        for likelihoods in out_net['likelihoods'].values()
    ).item()
    
    # Calcular el ratio de compresión
    compression_ratio = original_size / compressed_bits
    return compression_ratio 

# Nueva versión de la función para calcular el ratio de compresión
def compute_compression_ratio_from_h5_data(original_image, compressed_data):
    # Calcular el tamaño de la imagen original en bits
    original_size = original_image.numel() * original_image.element_size() * 8
    
    # Calcular el tamaño del archivo comprimido en bits
    # Sumamos los tamaños de y_hat y z_hat
    y_hat_bits = compressed_data['y_hat'].numel() * compressed_data['y_hat'].element_size() * 8
    z_hat_bits = compressed_data['z_hat'].numel() * compressed_data['z_hat'].element_size() * 8
    
    # Los valores mínimos y máximos, y las formas pueden ser opcionales
    meta_bits = 0
    if compressed_data['y_hat_min'] is not None:
        meta_bits += compressed_data['y_hat_min'].element_size() * 8
    if compressed_data['y_hat_max'] is not None:
        meta_bits += compressed_data['y_hat_max'].element_size() * 8
    if compressed_data['y_shape'] is not None:
        meta_bits += np.prod(compressed_data['y_shape']) * 8  # Tamaño en bits

    if compressed_data['z_hat_min'] is not None:
        meta_bits += compressed_data['z_hat_min'].element_size() * 8
    if compressed_data['z_hat_max'] is not None:
        meta_bits += compressed_data['z_hat_max'].element_size() * 8
    if compressed_data['z_shape'] is not None:
        meta_bits += np.prod(compressed_data['z_shape']) * 8  # Tamaño en bits
    
    # Tamaño total de la imagen comprimida
    compressed_bits = y_hat_bits + z_hat_bits + meta_bits

    # Calcular el ratio de compresión
    compression_ratio = original_size / compressed_bits
    return compression_ratio
