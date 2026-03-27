import h5py
import torch
import numpy as np

def save_compressed_data_to_h5(compressed_data, filename='compressed_image_reduced_sinlikelihoods_3.h5'):
    """
    Guarda los datos comprimidos en un archivo .h5 utilizando compresión.

    Args:
        compressed_data (dict): Diccionario que contiene los datos comprimidos y la información necesaria para la descompresión.
        filename (str): Nombre del archivo .h5 donde se guardarán los datos comprimidos.

    Returns:
        None
    """
    with h5py.File(filename, 'w') as hf:
        # Aplicar compresión solo a los arrays, no a los escalares
        hf.create_dataset('y_hat', data=compressed_data['y_hat'], dtype='uint8', compression='gzip', compression_opts=9)
        hf.create_dataset('y_hat_min', data=compressed_data['y_hat_min'])  # No compresión
        hf.create_dataset('y_hat_max', data=compressed_data['y_hat_max'])  # No compresión
        hf.create_dataset('y_shape', data=compressed_data['y_shape'])      # No compresión
        hf.create_dataset('z_hat', data=compressed_data['z_hat'], dtype='uint8', compression='gzip', compression_opts=9)
        hf.create_dataset('z_hat_min', data=compressed_data['z_hat_min'])  # No compresión
        hf.create_dataset('z_hat_max', data=compressed_data['z_hat_max'])  # No compresión
        hf.create_dataset('z_shape', data=compressed_data['z_shape'])      # No compresión

def save_compressed_data_to_h5_2(compressed_data, filename='compressed_image_reduced_sinlikelihoods_3.h5'):
    """
    Guarda los datos comprimidos en un archivo .h5 utilizando compresión donde sea aplicable.

    Args:
        compressed_data (dict): Diccionario que contiene los datos comprimidos y la información necesaria para la descompresión.
        filename (str): Nombre del archivo .h5 donde se guardarán los datos comprimidos.

    Returns:
        None
    """
    with h5py.File(filename, 'w') as hf:
        # Cuantización para optimizar los datos
        y_hat = np.clip(
            (compressed_data['y_hat'] - compressed_data['y_hat_min']) /
            (compressed_data['y_hat_max'] - compressed_data['y_hat_min']) * 255, 0, 255
        ).astype(np.uint8)

        z_hat = np.clip(
            (compressed_data['z_hat'] - compressed_data['z_hat_min']) /
            (compressed_data['z_hat_max'] - compressed_data['z_hat_min']) * 255, 0, 255
        ).astype(np.uint8)

        # Guardar los tensores cuantizados con compresión
        hf.create_dataset('y_hat', data=y_hat, compression='gzip', compression_opts=6)
        hf.create_dataset('z_hat', data=z_hat, compression='gzip', compression_opts=6)

        # Guardar los valores escalares sin compresión
        hf.create_dataset('y_hat_min', data=compressed_data['y_hat_min'])  # Sin compresión
        hf.create_dataset('y_hat_max', data=compressed_data['y_hat_max'])  # Sin compresión
        hf.create_dataset('y_shape', data=compressed_data['y_shape'])      # Sin compresión

        hf.create_dataset('z_hat_min', data=compressed_data['z_hat_min'])  # Sin compresión
        hf.create_dataset('z_hat_max', data=compressed_data['z_hat_max'])  # Sin compresión
        hf.create_dataset('z_shape', data=compressed_data['z_shape'])      # Sin compresión

def load_compressed_data_from_h5(filename, device='cpu'):
    with h5py.File(filename, 'r') as hf:
        y_hat_bytes = hf['y_hat'][:]
        y_hat_min = torch.tensor(hf['y_hat_min'][()], device=device)
        y_hat_max = torch.tensor(hf['y_hat_max'][()], device=device)
        y_shape = tuple(hf['y_shape'][:])
        z_hat_bytes = hf['z_hat'][:]
        z_hat_min = torch.tensor(hf['z_hat_min'][()], device=device)
        z_hat_max = torch.tensor(hf['z_hat_max'][()], device=device)
        z_shape = tuple(hf['z_shape'][:])

        print(f"Tamaño de y_hat_bytes leídos: {len(y_hat_bytes)} bytes")
        print(f"Tamaño de z_hat_bytes leídos: {len(z_hat_bytes)} bytes")

        # Convertir bytes a tensores
        y_hat_normalized = torch.tensor(np.frombuffer(y_hat_bytes.tobytes(), dtype=np.float32).reshape(y_shape)).to(device)
        z_hat_normalized = torch.tensor(np.frombuffer(z_hat_bytes.tobytes(), dtype=np.float32).reshape(z_shape)).to(device)

        # Desnormalizar los tensores
        y_hat = y_hat_normalized * (y_hat_max - y_hat_min) + y_hat_min
        z_hat = z_hat_normalized * (z_hat_max - z_hat_min) + z_hat_min

    return y_hat, z_hat

def load_compressed_data_from_h5_2(filename, device='cpu'):
    """
    Carga los datos comprimidos desde un archivo HDF5 y reconstruye los tensores originales.
    """
    with h5py.File(filename, 'r') as hf:
        # Leer los datos cuantizados
        y_hat_quantized = hf['y_hat'][:]
        z_hat_quantized = hf['z_hat'][:]

        # Leer escalares y formas
        y_hat_min = torch.tensor(hf['y_hat_min'][()], device=device)
        y_hat_max = torch.tensor(hf['y_hat_max'][()], device=device)
        y_shape = tuple(hf['y_shape'][:])

        z_hat_min = torch.tensor(hf['z_hat_min'][()], device=device)
        z_hat_max = torch.tensor(hf['z_hat_max'][()], device=device)
        z_shape = tuple(hf['z_shape'][:])

        # Verificar tamaños
        print(f"Tamaño real de y_hat: {y_hat_quantized.size}, Tamaño esperado: {np.prod(y_shape)}")
        print(f"Tamaño real de z_hat: {z_hat_quantized.size}, Tamaño esperado: {np.prod(z_shape)}")

        # Ajustar la forma de y_hat si no coincide
        if y_hat_quantized.size != np.prod(y_shape):
            print("Ajustando la forma de y_hat automáticamente...")
            possible_shapes_y = [(1, 128, 336, 300)]
            for shape in possible_shapes_y:
                if np.prod(shape) == y_hat_quantized.size:
                    y_shape = shape
                    print(f"Nueva forma encontrada para y_hat: {y_shape}")
                    break
            else:
                raise ValueError("No se puede ajustar y_hat: tamaño incompatible.")

        # Ajustar la forma de z_hat si no coincide
        if z_hat_quantized.size != np.prod(z_shape):
            print("Ajustando la forma de z_hat automáticamente...")
            possible_shapes_z = [(1, 128, 90, 70), (1, 128, 126, 50)]
            for shape in possible_shapes_z:
                if np.prod(shape) == z_hat_quantized.size:
                    z_shape = shape
                    print(f"Nueva forma encontrada para z_hat: {z_shape}")
                    break
            else:
                raise ValueError("No se puede ajustar z_hat: tamaño incompatible.")

        # Ajustar las formas
        y_hat_quantized = y_hat_quantized.reshape(y_shape)
        z_hat_quantized = z_hat_quantized.reshape(z_shape)

        # Convertir de uint8 a float32 y normalizar
        y_hat_normalized = torch.tensor(y_hat_quantized, dtype=torch.float32, device=device) / 255.0
        z_hat_normalized = torch.tensor(z_hat_quantized, dtype=torch.float32, device=device) / 255.0

        # Restaurar los valores originales
        y_hat = y_hat_normalized * (y_hat_max - y_hat_min) + y_hat_min
        z_hat = z_hat_normalized * (z_hat_max - z_hat_min) + z_hat_min

    return y_hat, z_hat

