import jax
import random
import numpy as np
from PIL import Image
import os

def preprocess_exemplar(image: Image, new_size: tuple[int, int]=(128, 128)):
    new_dimension = min(image.width, image.height)

    # Calculate the coordinates for the central crop
    left = (image.width - new_dimension) / 2
    top = (image.height - new_dimension) / 2
    right = (image.width + new_dimension) / 2
    bottom = (image.height + new_dimension) / 2

    # Crop the image to the central square
    cropped_image = image.crop((left, top, right, bottom))

    resized_image = cropped_image.resize(new_size)
    
    return resized_image