import torch
from src.utils.metrics import compute_bpp

def train_batch(model, batch, loss_fn, optimizer, device):
    model.train()
    x = batch.to(device)
    optimizer.zero_grad()
    output = model(x)
    x_hat = output['x_hat'] # Accedo a la reconstrucción de la imagen en el diccionario de salida
    loss = loss_fn(x, x_hat)
    if torch.isnan(loss):
        print("Error: Encontrado NaN en la pérdida.")
        return None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def train_batch_optuna(model, batch, loss_fn, optimizer, device, lambda_value):
    model.train()
    x = batch.to(device)
    optimizer.zero_grad()
    output = model(x)
    x_hat = output['x_hat'] # Accedo a la reconstrucción de la imagen en el diccionario de salida
    #bit_rate = output['bpp']  # Asumiendo que 'bpp' es la tasa de bits
    bit_rate = compute_bpp(output)  # Asumiendo que 'bpp' es la tasa de bits

    loss = loss_fn(x, x_hat, bit_rate, lambda_value)  # Pasar lambda_value a la función de pérdida
    if torch.isnan(loss):
        print("Error: Encontrado NaN en la pérdida.")
        return None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()
