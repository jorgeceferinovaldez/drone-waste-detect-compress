import torch
import numpy as np

def compress_image(net, x):
    """
    Realiza la inferencia y compresión de la imagen utilizando el modelo `compressai`.

    Args:
        net: El modelo de compresión (e.g., `cheng2020_anchor`).
        x: El tensor de entrada de la imagen.

    Returns:
        dict: Un diccionario que contiene los datos comprimidos y la información necesaria para la descompresión.
    """
    with torch.no_grad():
        y = net.g_a(x)
        y_hat, y_likelihoods = net.entropy_bottleneck(y)
        z = net.h_a(y)
        z_hat, z_likelihoods = net.entropy_bottleneck(z)

        # Normalización de los tensores
        y_hat_min, y_hat_max = y_hat.min(), y_hat.max()
        z_hat_min, z_hat_max = z_hat.min(), z_hat.max()

        y_hat_normalized = (y_hat - y_hat_min) / (y_hat_max - y_hat_min)
        z_hat_normalized = (z_hat - z_hat_min) / (z_hat_max - z_hat_min)

        # Convertir los tensores normalizados a bytes para ocupar menos espacio
        y_hat_bytes = np.frombuffer(y_hat_normalized.cpu().numpy().tobytes(), dtype=np.uint8)
        z_hat_bytes = np.frombuffer(z_hat_normalized.cpu().numpy().tobytes(), dtype=np.uint8)

        #print(f"Tamaño de y_hat_bytes: {len(y_hat_bytes)} bytes")
        #print(f"Tamaño de z_hat_bytes: {len(z_hat_bytes)} bytes")

        compressed_data = {
            'y_hat': y_hat_bytes,
            'y_hat_min': y_hat_min.cpu().numpy(),
            'y_hat_max': y_hat_max.cpu().numpy(),
            'y_shape': y_hat.shape,
            'z_hat': z_hat_bytes,
            'z_hat_min': z_hat_min.cpu().numpy(),
            'z_hat_max': z_hat_max.cpu().numpy(),
            'z_shape': z_hat.shape,
        }

    return compressed_data

def decompress_image(net, y_hat):
    """
    Reconstruye la imagen a partir de los tensores desnormalizados usando el modelo `compressai`.

    Args:
        net: El modelo de compresión (e.g., `cheng2020_anchor`).
        y_hat: El tensor desnormalizado que contiene la información comprimida de la imagen.

    Returns:
        torch.Tensor: La imagen reconstruida, con valores de píxel en el rango [0, 1].
    """
    with torch.no_grad():
        y_hat = net.entropy_bottleneck.dequantize(y_hat)
        x_hat = net.g_s(y_hat).clamp(0, 1)

    return x_hat


