from abc import ABC, abstractmethod
from tensorflow import keras


class ModelBuilder(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes):
        pass


class SampleCNNBuilder(ModelBuilder):
    def __init__(self, n_filters=32) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'SampleCNN'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model