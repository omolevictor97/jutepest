### This module takes in either a file path and preprocessed
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from PIL import Image
from src.constant import *

def preprocess_img(input_data, target_size=IMAGE_SIZE):
    # Load the image using Keras' load_img function
    """
    This function preprocesses an image from either a file path or a PIL/Image
    """
    try:
        if not os.path.exists(input_data):
            raise FileNotFoundError(f'File not found: {input_data}')
        else:
            img = load_img(input_data, target_size=target_size)
            img_arr = img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            return img_arr
    except Exception as e:
        return str(e)
    
