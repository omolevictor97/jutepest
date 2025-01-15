### This file basically contains the running of our application

import os
from src.constant import *
from src.data_ingest import DataIngest
from src.data_pipeline import DataLoader
from src.components.train import Trainer
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

def run_model():
    data_ingester = DataIngest(data_folder= DATA_FOLDER_PATH)
    data_dir = data_ingester.read_data_path()
    data_loader = DataLoader(data_dir=data_dir)
    train_ds, val_ds, test_ds, class_names = data_loader.load_data()
    output_unit = len(class_names)
    trainer = Trainer(train_ds=train_ds, val_ds=val_ds, output_unit=output_unit)
    model = trainer.train()
    return model, class_names


def save_model():
    model, class_names = run_model()

    if model:
        model_dir = os.path.join(os.getcwd(), MODEL_DIR_PATH)
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, 'jutemodel.keras'))
        print(f"Model saved at {model_dir}")
        print(F'Classes found are: {class_names}')

def list_folder_names(DATA_FOLDER_PATH):
    class_names = [folder for folder in os.listdir(os.path.join(DATA_FOLDER_PATH, 'train'))]
    return class_names

def predict(model_path, img_arr, class_names):
    # load the model
    model = load_model(model_path)
    # make prediction
    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return class_names[np.argmax(score)], round(100 * np.max(score), 2)

if __name__ == '__main__':
    save_model()


