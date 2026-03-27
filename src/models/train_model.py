import torch

import sys
from pathlib import Path
import os

# Me aseguro de que el directorio raíz del proyecto esté en el sys.path
project_root = Path(os.path.abspath("")).parent

# Añado el directorio raíz al sys.path si no está ya presente
if project_root not in sys.path:
    sys.path.append(str(project_root))

# Importo las funciones de configuración
from src.config import load_config
from src.utils.metrics import calculate_metrics



def train_batch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for blur_images, original_image in train_loader:
        blur_images = blur_images.to(device)
        original_image = original_image.to(device)
        
        optimizer.zero_grad()
        outputs = model(blur_images)
        
        original_image_expanded = original_image.unsqueeze(1).expand_as(outputs)
        
        loss = criterion(outputs, original_image_expanded)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    return train_loss

@torch.no_grad()
def compute_val_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    for blur_images, original_image in val_loader:
        blur_images = blur_images.to(device)
        original_image = original_image.to(device)
        
        outputs = model(blur_images)
        
        original_image_expanded = original_image.unsqueeze(1).expand_as(outputs)
        
        loss = criterion(outputs, original_image_expanded)
        
        val_loss += loss.item()
        
        # Convertir las imágenes a numpy para calcular las métricas
        original_np = original_image.cpu().numpy().transpose(0, 2, 3, 1)
        decompressed_np = outputs.cpu().detach().numpy().reshape(-1, 3, 256, 256).transpose(0, 2, 3, 1)  # Ajustar tamaño de 512 a 256
        
        for orig, dec in zip(original_np, decompressed_np):
            psnr_value, ssim_value = calculate_metrics(orig, dec)
            val_psnr += psnr_value
            val_ssim += ssim_value

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader.dataset)
    val_ssim /= len(val_loader.dataset)
    return val_loss, val_psnr, val_ssim

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, output_model_path, nombre_modelo, early_stop_patience):
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    val_psnr_values = []
    val_ssim_values = []

    for epoch in range(num_epochs):
        train_loss = train_batch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss, val_psnr, val_ssim = compute_val_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_psnr_values.append(val_psnr)
        val_ssim_values.append(val_ssim)

        # Guardar los valores en tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('val/PSNR', val_psnr, epoch)
        writer.add_scalar('val/SSIM', val_ssim, epoch)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_model_path}/{nombre_modelo}.pth')
            early_stop_counter = 0  # Reset counter if we get a new best validation loss
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping activated. Training stopped at epoch {epoch+1}')
            break

    writer.close() # Cerrar tensorboard
    return train_losses, val_losses, val_psnr_values, val_ssim_values