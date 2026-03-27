import torch

import sys
from pathlib import Path
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Me aseguro de que el directorio raíz del proyecto esté en el sys.path
project_root = Path(os.path.abspath("")).parent

# Añado el directorio raíz al sys.path si no está ya presente
if project_root not in sys.path:
    sys.path.append(str(project_root))


from src.config import processed_data_dir, models_dir, reports_dir, load_config
from src.utils.metrics import calculate_metrics, calculate_compression_ratio

# @torch.no_grad()
# def compute_val_metrics(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0.0
#     val_psnr = 0.0
#     val_ssim = 0.0
#     compression_ratios = []
#     #for original_image, _ in val_loader:
#     for original_image in val_loader:
#         original_image = original_image.to(device)
        
#         encoded = model.compress(original_image)
#         outputs = model.decompress(encoded)
        
#         loss = criterion(outputs, original_image)
        
#         val_loss += loss.item()
        
#         # Convertir las imágenes a numpy para calcular las métricas
#         original_np = original_image.cpu().numpy().transpose(0, 2, 3, 1)
#         reconstructed_np = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)
        
#         for orig, rec in zip(original_np, reconstructed_np):
#             psnr_value, ssim_value = calculate_metrics(orig, rec, device)
#             val_psnr += psnr_value
#             val_ssim += ssim_value

#         # Calcular la razón de compresión
#         batch_size, latent_channels, latent_height, latent_width = encoded.size()
#         latent_size = latent_height * latent_width * latent_channels
#         original_size = 256 * 256 * 3
#         compression_ratio = calculate_compression_ratio(original_size, latent_size)
#         compression_ratios.append(compression_ratio)

#     val_loss /= len(val_loader)
#     val_psnr /= len(val_loader.dataset)
#     val_ssim /= len(val_loader.dataset)
#     avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    
#     return val_loss, val_psnr, val_ssim, avg_compression_ratio

@torch.no_grad()
def compute_val_metrics(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    compression_ratios = []

    for original_image in val_loader:
        original_image = original_image.to(device)
        
        encoded = model.compress(original_image)
        outputs = model.decompress(encoded)
        
        loss = criterion(outputs, original_image)
        val_loss += loss.item()
        
        # Pasar directamente los tensores a calculate_metrics
        for orig, rec in zip(original_image, outputs):
            psnr_value, ssim_value = calculate_metrics(orig, rec, device)
            val_psnr += psnr_value
            val_ssim += ssim_value

        # Calcular la razón de compresión
        batch_size, latent_channels, latent_height, latent_width = encoded.size()
        latent_size = latent_height * latent_width * latent_channels
        original_size = 256 * 256 * 3
        compression_ratio = calculate_compression_ratio(original_size, latent_size)
        compression_ratios.append(compression_ratio)

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader.dataset)
    val_ssim /= len(val_loader.dataset)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    
    return val_loss, val_psnr, val_ssim, avg_compression_ratio
