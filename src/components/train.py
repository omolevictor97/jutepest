from src.constant import *
from src.data_ingest import *
from src.data_pipeline import *
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential


class Trainer:
    def __init__(self, train_ds, val_ds, output_unit, epochs:int=EPOCH):
        self.base_model = EfficientNetV2B1(include_top = False, weights = 'imagenet', input_shape = IMAGE_SIZE + (3,))
        self.base_model.trainable = False
        self.model = None
        self.output_unit  = output_unit
        self.epochs = epochs
        self.train_ds = train_ds
        self.val_ds = val_ds

    def train(self):
        self.model = Sequential(
            [
                self.base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.output_unit, activation='sigmoid')
            ]
        )

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        self.model.fit(
            self.train_ds,
            validation_data = self.val_ds,
            epochs = self.epochs
        )

        print(self.model.summary())
        return self.model