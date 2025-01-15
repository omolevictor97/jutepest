from preprocess import preprocess_img
from tensorflow.keras.models import load_model
import tensorflow as tf
import glob
import os
from src.constant import *
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from utils import list_folder_names, predict
import streamlit as st


model_dir = os.path.join(os.getcwd(), MODEL_DIR_PATH, 'jutemodel.keras')
class_names = list_folder_names(DATA_FOLDER_PATH=DATA_FOLDER_PATH)

def preprocess(img):
    test_img = load_img(img, target_size=IMAGE_SIZE)
    test_img = img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    return test_img


def main():
    st.title('JUTE PEST DETECTION')

    uploaded_image = st.file_uploader('Select an Image....', type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)
        image_arr = preprocess(uploaded_image)
        result = predict(
            model_path=model_dir,
            img_arr=image_arr,
            class_names=class_names
        )

        st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(result[0], result[1])
    )
if __name__ == '__main__':
    # img_path = os.path.normpath('Beet Armyworm\Image_1_jpg.rf.8628e8e78212aa6b37714ba057287e64.jpg')
    # img_path = os.path.join(DATA_FOLDER_PATH, 'train', img_path)
    # img_arr = preprocess_img(input_data=img_path)
    # result = predict(model_path=model_dir, img_arr=img_arr, class_names=class_names)
    # print(result)
    main()