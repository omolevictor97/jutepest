import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from src.constant import *
from src.data_ingest import DataIngest


class DataLoader:
    def __init__(self, data_dir : tuple):
        self.train_path = data_dir[0]
        self.val_path = data_dir[1]
        self.test_path = data_dir[2]

    def load_data(self, image_size : tuple = IMAGE_SIZE, batch_size : int = BATCH_SIZE):
        try:
            train_ds = image_dataset_from_directory(
                directory = self.train_path,
                color_mode = 'rgb',
                shuffle = True,
                image_size = image_size,
                batch_size = BATCH_SIZE
            )

            val_ds = image_dataset_from_directory(
                directory = self.val_path,
                color_mode = 'rgb',
                batch_size = BATCH_SIZE,
                image_size = image_size
            )

            test_ds = image_dataset_from_directory(
                directory = self.test_path,
                color_mode = 'rgb',
                batch_size = BATCH_SIZE,
                image_size = image_size
            )
            class_names = train_ds.class_names

            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

            return train_ds, val_ds, test_ds, class_names
        except Exception as E:
            return str(E)


### To Test run the application
