import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom Dataset class
# class CustomDataset(Dataset):
#     def __init__(self, original_dir, blur_dir, transform=None):
#         self.original_dir = original_dir
#         self.blur_dir = blur_dir
#         self.transform = transform
#         self.image_list = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         original_img_name = self.image_list[idx]
#         original_img_path = os.path.join(self.original_dir, original_img_name)
#         original_image = Image.open(original_img_path).convert("RGB")

#         base_name = original_img_name.rsplit('.', 1)[0]  # Remuevo la extensión
#         blur_images = []
#         for i in range(10):
#             blur_img_name = f"{base_name}_{i}.png"  # Cambio a .png
#             blur_img_path = os.path.join(self.blur_dir, blur_img_name)
#             if os.path.exists(blur_img_path):
#                 blur_image = Image.open(blur_img_path).convert("RGB")
#                 if self.transform:
#                     blur_image = self.transform(blur_image)
#                 blur_images.append(blur_image)
#             else:
#                 raise FileNotFoundError(f"File {blur_img_path} not found")

#         if self.transform:
#             original_image = self.transform(original_image)
        
#         # Stack the blur images into a single tensor
#         # Estackeo las imágenes borrosas en un solo tensor
#         # (10, 3, 256, 256) -> (10, 256, 256, 3) -> (10, 3, 256, 256)
#         blur_images_tensor = torch.stack(blur_images)
#         return blur_images_tensor, original_image


# Dataset personalizado
# class CustomDataset(Dataset):
#     def __init__(self, original_dir, transform=None):
#         self.original_dir = original_dir
#         self.transform = transform
#         self.image_names = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self, idx):
#         original_image = Image.open(os.path.join(self.original_dir, self.image_names[idx]))
        
#         if self.transform:
#             original_image = self.transform(original_image)
        
#         return original_image, original_image
    
class CustomDataset(Dataset):
    def __init__(self, original_dir, transform=None):
        self.original_dir = original_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        original_image = Image.open(os.path.join(self.original_dir, self.image_names[idx]))
        if self.transform:
            original_image = self.transform(original_image)
        return original_image
    
class CustomDataset_2(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Leer el archivo CSV
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # Obtener la ruta de la imagen
        label = self.data.iloc[idx, 1]  # Obtener la etiqueta
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        #return image, label  # Retornar la imagen y la etiqueta
        return image
    
class CustomDataset_2_2(Dataset):
    def __init__(self, csv_file, transform=None, augmentation_pipeline=None, use_augmentation=False):
        self.data = pd.read_csv(csv_file)  # Leer el archivo CSV
        self.transform = transform
        self.augmentation_pipeline = augmentation_pipeline
        self.use_augmentation = use_augmentation  # Parámetro para controlar si usar Data Augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # Obtener la ruta de la imagen
        image = Image.open(img_path)
        image = np.array(image)  # Convertir la imagen a un array de NumPy

        # Aplicar Data Augmentation solo si use_augmentation es True
        if self.use_augmentation and self.augmentation_pipeline:
            image = self.augmentation_pipeline(image=image)['image']

        # Convertir de nuevo a PIL para las transformaciones de torchvision
        image = Image.fromarray(image)

        # Aplicar las transformaciones si están definidas
        if self.transform:
            image = self.transform(image)

        return image  # Retornar solo la imagen transformada o aumentada
    

