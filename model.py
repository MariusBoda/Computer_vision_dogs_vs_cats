from collections import defaultdict
from glob import iglob
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt


import arrow
import base64
import os
from PIL import Image
import pandas as pd
from glob import glob
from io import BytesIO
from os.path import basename

DEVICE = torch.device('cpu')
OUTPUT_SIZE = 2048

#model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

cats = Path("C:/Users/mariu/Documents/House-Plant-Species-Computer-Vision-Prediction/data/cat")
dogs = Path("C:/Users/mariu/Documents/House-Plant-Species-Computer-Vision-Prediction/data/dog")

# Get lists of PNG images
cat_images = list(cats.glob("*.png"))
dog_images = list(dogs.glob("*.png"))

# Combine lists
IMAGE_PATH_LIST = cat_images + dog_images

images_path = [None] * len(IMAGE_PATH_LIST)  
labels = [None] * len(IMAGE_PATH_LIST)

for i,img_path in enumerate(IMAGE_PATH_LIST):
    images_path[i] = img_path
    labels[i] = img_path.parent.stem
    
dataset_df = pd.DataFrame({'image_path':images_path, 
                                  'label':labels})

def _load(image_path, as_tensor=True):

    image = Image.open(image_path)
    
    if as_tensor:          # Convert the PIL Image to a PyTorch tensor using torchvision's ToTensor transform.
        converter = transforms.ToTensor()    
        return converter(image)
    else:                  # Return the PIL Image object without conversion.
        return image  

def view_multiple_samples(df, sample_loader, count=10, color_map='rgb', fig_size=(14, 10)):
    rows = count // 5
    if count % 5 > 0:
        rows += 1
    
    idx = random.sample(df.index.to_list(), count)
    fig = plt.figure(figsize=fig_size)

    for column, _ in enumerate(idx):
        plt.subplot(rows, 5, column + 1)
        plt.title(f'Label: {df.label[_]}')
        
        if color_map == 'rgb':
            plt.imshow(sample_loader(df.image_path[_]).permute(1, 2, 0))
        else:
            plt.imshow(sample_loader(df.image_path[_]).permute(1, 2, 0), cmap=color_map)
        
        plt.axis('off')  # Optionally turn off axes for a cleaner look

    plt.show()  # Ensure the plot is displayed


# Call view_mulitiple_samples function to display 20 random sample images.
view_multiple_samples(
    dataset_df, _load, 
    count=20, fig_size=(20, 24)  # View 20 random sample images
)