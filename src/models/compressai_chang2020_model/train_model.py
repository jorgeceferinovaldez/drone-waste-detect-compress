import torch

import sys
from pathlib import Path
import os

# Me aseguro de que el directorio raíz del proyecto esté en el sys.path
project_root = Path(os.path.abspath("")).parent

# Añado el directorio raíz al sys.path si no está ya presente
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.models.compressai_chang2020_model.train_batch import train_batch, train_batch_optuna
from src.models.compressai_chang2020_model.validate_batch import compute_val_metrics, compute_val_metrics_optuna

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, output_model_path, nombre_modelo, num_epochs, device, writer):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnr_values = []
    val_ssim_values = []
    compression_ratios = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch in train_loader:
            batch_loss = train_batch(model, batch, loss_fn, optimizer, device)

            if batch_loss is not None:
                train_loss += batch_loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss, val_psnr, val_ssim, avg_compression_ratio = compute_val_metrics(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        val_psnr_values.append(val_psnr)
        val_ssim_values.append(val_ssim)
        compression_ratios.append(avg_compression_ratio)

        # Guardar los valores en tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('val/PSNR', val_psnr, epoch)
        writer.add_scalar('val/SSIM', val_ssim, epoch)
        writer.add_scalar('val/Compression_Ratio', avg_compression_ratio, epoch)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}, Compression Ratio: {avg_compression_ratio:.2f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_model_path}/{nombre_modelo}.pth')
            #early_stop_counter = 0  # Reset counter if we get a new best validation loss
        else:
            #early_stop_counter += 1
            pass
    
    writer.close() # Cerrar tensorboard
    return train_losses, val_losses, val_psnr_values, val_ssim_values, compression_ratios

# Funcion crada para usar Optuna
def train_model_optuna(model, train_loader, val_loader, loss_fn, optimizer, scheduler, output_model_path, nombre_modelo, num_epochs, device, writer, lambda_value):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_psnr_values = []
    val_ssim_values = []
    compression_ratios = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch in train_loader:
            batch_loss = train_batch_optuna(model, batch, loss_fn, optimizer, device, lambda_value)

            if batch_loss is not None:
                train_loss += batch_loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss, val_psnr, val_ssim, avg_compression_ratio = compute_val_metrics_optuna(model, val_loader, loss_fn, device, lambda_value)
        val_losses.append(val_loss)
        val_psnr_values.append(val_psnr)
        val_ssim_values.append(val_ssim)
        compression_ratios.append(avg_compression_ratio)

        # Guardar los valores en tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('val/PSNR', val_psnr, epoch)
        writer.add_scalar('val/SSIM', val_ssim, epoch)
        writer.add_scalar('val/Compression_Ratio', avg_compression_ratio, epoch)

        scheduler.step(val_loss)

        # Monitorear la tasa de aprendizaje
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}], Current learning rate: {current_lr}")

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}, Compression Ratio: {avg_compression_ratio:.2f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_model_path}/{nombre_modelo}.pth')
            #early_stop_counter = 0  # Reset counter if we get a new best validation loss
        else:
            #early_stop_counter += 1
            pass
    
    writer.close() # Cerrar tensorboard
    return train_losses, val_losses, val_psnr_values, val_ssim_values, compression_ratios