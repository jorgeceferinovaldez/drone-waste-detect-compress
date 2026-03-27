import os
from pathlib import Path
from src.config import raw_data_dir, load_config

# Función para descargar archivos de Google Drive
def download_file_from_google_drive(url, destination):
    import requests
    session = requests.Session()
    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

# Función para obtener el token de confirmación
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# Función para guardar el contenido de la respuesta
def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filtra los chunks que están vacíos
                f.write(chunk)

# Descargar el archivo de Google Drive
if __name__ == "__main__":
    config = load_config() # Cargar configuración desde el archivo config.yaml

    file_id = config['data']['google_drive_file_id']
    destination = raw_data_dir() / 'dataset.zip'

    if not raw_data_dir().exists():
        raw_data_dir().mkdir(parents=True)

    download_file_from_google_drive(f'https://drive.google.com/uc?export=download&id={file_id}', destination)

    # Descomprimir el archivo si es necesario
    import zipfile
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir())

    # Elimina el archivo zip después de descomprimir
    os.remove(destination)

