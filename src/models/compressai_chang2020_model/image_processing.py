import math
from PIL import Image

# Función para ajustar el tamaño de la imagen
def pad_image_to_multiple(image, multiple=64):
    width, height = image.size
    new_width = math.ceil(width / multiple) * multiple
    new_height = math.ceil(height / multiple) * multiple
    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, (0, 0))
    return padded_image