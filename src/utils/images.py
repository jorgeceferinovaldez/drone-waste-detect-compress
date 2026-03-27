import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


# Display images
def display_images(original, compressed_channel, decompressed):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(compressed_channel, cmap='gray')
    axes[1].set_title('Compressed Channel')
    axes[1].axis('off')

    axes[2].imshow(decompressed)
    axes[2].set_title('Decompressed Image')
    axes[2].axis('off')

    plt.show()


def convertir_jpg_a_png(input_directory, output_directory, max_workers=None):
    """
    Converts all JPEG images in the input directory to PNG format and saves them in the output directory.

    Args:
        input_directory (str): The path to the directory containing the JPEG images.
        output_directory (str): The path to the directory where the converted PNG images will be saved.
        max_workers (int, optional): The maximum number of threads to use. If None, it uses the number of available processors.

    Returns:
        None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def convertir_imagen(file):
        if file.endswith('.jpg'):
            filepath = os.path.join(root, file)
            img = Image.open(filepath)
            filename = os.path.splitext(file)[0]
            output_path = os.path.join(output_directory, f"{filename}.png")
            img.save(output_path, 'PNG')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(input_directory):
            executor.map(convertir_imagen, files)

def convertir_jpg_a_png_2(input_directory, output_directory, max_workers=None):
    """
    Converts all JPEG images in the input directory and its subdirectories to PNG format 
    and saves them in the output directory, preserving the directory structure.

    Args:
        input_directory (str): The path to the directory containing the JPEG images.
        output_directory (str): The path to the directory where the converted PNG images will be saved.
        max_workers (int, optional): The maximum number of threads to use. If None, it uses the number of available processors.

    Returns:
        None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def convertir_imagen(filepath, output_subdir):
        if filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
            img = Image.open(filepath)
            filename = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(output_subdir, f"{filename}.png")
            img.save(output_path, 'PNG')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(input_directory):
            relative_path = os.path.relpath(root, input_directory)
            output_subdir = os.path.join(output_directory, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Crear tareas para cada imagen
            for file in files:
                filepath = os.path.join(root, file)
                executor.submit(convertir_imagen, filepath, output_subdir)

def copiar_png_existente(input_directory, output_directory, max_workers=None):
    """
    Copy existing PNG files from the input directory to the output directory.

    Args:
        input_directory (str): The path to the input directory.
        output_directory (str): The path to the output directory.
        max_workers (int, optional): The maximum number of threads to use. If None, it uses the number of available processors.

    Returns:
        None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def copiar_archivo(file):
        if file.endswith('.png'):
            filepath = os.path.join(root, file)
            output_path = os.path.join(output_directory, file)
            shutil.copy(filepath, output_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(input_directory):
            executor.map(copiar_archivo, files)

def copiar_png_existente_2(input_directory, output_directory, max_workers=None):
    """
    Copy existing PNG files from the input directory and its subdirectories to the output directory,
    preserving the directory structure.

    Args:
        input_directory (str): The path to the input directory.
        output_directory (str): The path to the output directory.
        max_workers (int, optional): The maximum number of threads to use. If None, it uses the number of available processors.

    Returns:
        None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def copiar_archivo(filepath, output_subdir):
        if filepath.endswith('.png'):
            output_path = os.path.join(output_subdir, os.path.basename(filepath))
            shutil.copy(filepath, output_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(input_directory):
            relative_path = os.path.relpath(root, input_directory)
            output_subdir = os.path.join(output_directory, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            for file in files:
                filepath = os.path.join(root, file)
                executor.submit(copiar_archivo, filepath, output_subdir)

def tenengrad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(sobelx, sobely)
    return np.mean(gradient_magnitude)


def classify_images_with_quartile_threshold(input_dir, output_dir_focused, output_dir_unfocused, threshold, max_workers=None):
    if not os.path.exists(output_dir_focused):
        os.makedirs(output_dir_focused) 
    if not os.path.exists(output_dir_unfocused):
        os.makedirs(output_dir_unfocused)

    def process_image(filename):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            tenengrad_score = tenengrad(image)
            
            if tenengrad_score < threshold:
                shutil.copy(image_path, os.path.join(output_dir_unfocused, filename))
                print(f'{filename} está desenfocada, puntuación: {tenengrad_score}')
            else:
                shutil.copy(image_path, os.path.join(output_dir_focused, filename))
                print(f'{filename} está bien enfocada, puntuación: {tenengrad_score}')

    filenames = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_image, filenames)

def calculate_scores(input_dir, max_workers=None):
    def process_image(filename):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            tenengrad_score = tenengrad(image)
            return tenengrad_score, filename

    filenames = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    scores = []
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, filenames))

    scores, filenames = zip(*results) if results else ([], [])

    return list(scores), list(filenames)