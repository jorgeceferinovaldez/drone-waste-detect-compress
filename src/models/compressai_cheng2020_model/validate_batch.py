import torch

import sys
from pathlib import Path
import os

# Importar desde torchmetrics.image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Me aseguro de que el directorio raíz del proyecto esté en el sys.path
project_root = Path(os.path.abspath("")).parent

# Añado el directorio raíz al sys.path si no está ya presente
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.utils.metrics import compute_compression_ratio, compute_bpp

@torch.no_grad()
def compute_val_metrics(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    compression_ratios = []

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    for batch in val_loader:
        x = batch.to(device)
        output = model(x)
        x_hat = output['x_hat']
        loss = loss_fn(x, x_hat)
        val_loss += loss.item()

        psnr_value = psnr_metric(x, x_hat)
        ssim_value = ssim_metric(x, x_hat)
        compression_ratio = compute_compression_ratio(output, x)
        compression_ratios.append(compression_ratio)

        val_psnr += psnr_value
        val_ssim += ssim_value

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    return val_loss, val_psnr, val_ssim, avg_compression_ratio

@torch.no_grad()
def compute_val_metrics_optuna(model, val_loader, loss_fn, device, lambda_value):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    compression_ratios = []

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    for batch in val_loader:
        x = batch.to(device)
        output = model(x)
        x_hat = output['x_hat']
        #bit_rate = output['bpp']  # Asumiendo que 'bpp' es la tasa de bits
        bit_rate = compute_bpp(output)  # Asumiendo que 'bpp' es la tasa de bits

        loss = loss_fn(x, x_hat, bit_rate, lambda_value)  # Pasar lambda_value a la función de pérdida
        val_loss += loss.item()

        psnr_value = psnr_metric(x, x_hat)
        ssim_value = ssim_metric(x, x_hat)
        compression_ratio = compute_compression_ratio(output, x)
        compression_ratios.append(compression_ratio)

        val_psnr += psnr_value
        val_ssim += ssim_value

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    return val_loss, val_psnr, val_ssim, avg_compression_ratio
