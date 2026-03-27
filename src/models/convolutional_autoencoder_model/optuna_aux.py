import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import sys
from pathlib import Path
import os


import optuna

from torch.utils.tensorboard import SummaryWriter

# Me aseguro de que el directorio raíz del proyecto esté en el sys.path
project_root = Path(os.path.abspath("")).parent

# Añado el directorio raíz al sys.path si no está ya presente
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.config import processed_data_dir, reports_dir, load_config

from src.models.convolutional_autoencoder_model.model import ConvolutionalAutoencoder

from src.models.convolutional_autoencoder_model.train_batch import train_batch
from src.models.convolutional_autoencoder_model.validate_batch import compute_val_metrics

# Defino la función objetivo para Optuna
def objective(trial, train_dataset, val_dataset, device):
    # Defino los hiperparámetros a optimizar
    encoder_filters = [64, 128, 256, 512, 1024, 2048]
    decoder_filters = list(reversed(encoder_filters))
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 128, step=8) 
        
    # Imprimo los hiperparámetros usados en el trial actual
    print(f"Trial {trial.number}:")
    print(f"  encoder_filters: {encoder_filters}")
    print(f"  decoder_filters: {decoder_filters}")
    print(f"  lr: {lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  batch_size: {batch_size}")

    # Defino el modelo
    model = ConvolutionalAutoencoder(encoder_filters, decoder_filters).to(device)
    
    # Defino el criterio y el optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Crear los dataloaders con el batch_size actual
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Inicializar TensorBoard
    nombre_modelo = 'convolutional_autoencoder_model_optuna'
    log_dir = reports_dir() / "logs"/ nombre_modelo / f"trial_{trial.number}"
    print(f"Guardando logs en {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Entrenar el modelo
    num_epochs = 30
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 8
    train_losses = []
    val_losses = []
    val_psnr_values = []
    val_ssim_values = []
    compression_ratios = []
    
    for epoch in range(num_epochs):
        train_loss = train_batch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        val_loss, val_psnr, val_ssim, avg_compression_ratio = compute_val_metrics(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_psnr_values.append(val_psnr)
        val_ssim_values.append(val_ssim)
        compression_ratios.append(avg_compression_ratio)
        
        # Guardar los valores en TensorBoard y imprimir
        writer.add_scalar(f'trial_{trial.number}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'trial_{trial.number}/Loss/val', val_loss, epoch)
        writer.add_scalar(f'trial_{trial.number}/val/PSNR', val_psnr, epoch)
        writer.add_scalar(f'trial_{trial.number}/val/SSIM', val_ssim, epoch)
        writer.add_scalar(f'trial_{trial.number}/val/Compression_Ratio', avg_compression_ratio, epoch)

        scheduler.step(val_loss)

        # Registrar histogramas de los parámetros del modelo
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}', param, epoch)

        print(f'Trial {trial.number}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}, Compression Ratio: {avg_compression_ratio:.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            break
    # Justo antes de writer.close()
    hparams = {
        "encoder_filters": str(encoder_filters),  # Convertir a str
        "decoder_filters": str(decoder_filters),  # Convertir a str
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "trial_number": trial.number,
    }

    metrics = {
        "best_val_loss": best_val_loss,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "avg_compression_ratio": avg_compression_ratio,
    }

    writer.add_hparams(hparams, metrics)
    writer.close()  # Cerrar TensorBoard
    return best_val_loss