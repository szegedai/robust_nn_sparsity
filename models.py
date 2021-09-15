from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from foolbox.attacks import LinfPGD
from foolbox.models import TensorFlowModel


class SplitFlatten(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SplitFlatten, self).__init__(trainable=False, **kwargs)
        self.axis = axis

    def get_config(self):
        return {"axis": self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, **kwargs):
        return tf.reshape(tf.stack(tf.split(inputs, 2, axis=self.axis)), [-1, tf.math.reduce_prod(inputs.shape[1:])])


class ModelBuilder(ABC):
    @abstractmethod
    def build_model(self, input_shape, nb_classes):
        pass

    @abstractmethod
    def get_name(self):
        pass


"""class VGG4(tf.keras.models.Model):
    def __init__(self, input_shape, n_classes, width_scale=1):
        super(VGG4, self).__init__()
        self.width_scale = width_scale

        self.conv_2d_0 = tf.keras.layers.Conv2D(32 * width_scale, (5, 5), padding="same", input_shape=input_shape, activation="relu")
        self.max_pooling_2d_0 = tf.keras.layers.MaxPooling2D()
        self.conv_2d_1 = tf.keras.layers.Conv2D(64 * width_scale, (5, 5), padding="same", activation="relu")
        self.max_pooling_2d_1 = tf.keras.layers.MaxPooling2D()
        self.flatten_0 = tf.keras.layers.Flatten()
        self.dense_0 = tf.keras.layers.Dense(1024 * width_scale, activation="relu")
        self.dense_1 = tf.keras.layers.Dense(n_classes)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        x = self.conv_2d_0(inputs)
        x = self.max_pooling_2d_0(x)
        x = self.conv_2d_1(x)
        x = self.max_pooling_2d_1(x)
        x = self.flatten_0(x)
        x = self.dense_0(x)
        return self.dense_1(x)

    @property
    def name(self):
        return f"VGG4_ws{self.width_scale}"
"""


@tf.function
def PGD_attack(model, x, y, step_size, steps, eps, random_start=True, valid_value_range=(0.0, 1.0)):
    adv_x = tf.identity(x)
    if random_start:
        adv_x = adv_x + tf.random.uniform(tf.shape(adv_x), minval=-eps, maxval=eps)
    for _ in range(steps):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(adv_x)
            predictions = model(adv_x, training=False)
            loss = model.compiled_loss(y, predictions)
            grads = tape.gradient(loss, adv_x)
        #tf.print(grads)
        adv_x = adv_x + step_size * tf.sign(grads)
        adv_x = tf.clip_by_value(adv_x, x - eps, x + eps)
        adv_x = tf.clip_by_value(adv_x, *valid_value_range)
    #tf.print(tf.norm(adv_x - x, ord=np.inf))
    return adv_x, y


class AdversarialModel(tf.keras.models.Model):
    def __init__(self, step_size, steps, eps, valid_value_range=(0.0, 1.0), **kwargs):
        super(AdversarialModel, self).__init__(**kwargs)
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.valid_value_range = valid_value_range

    @tf.function
    def train_step(self, data):
        return super(AdversarialModel, self).train_step(PGD_attack(self, *data, self.step_size, self.steps, self.eps))

    @tf.function
    def test_step(self, data):
        return super(AdversarialModel, self).test_step(PGD_attack(self, *data, self.step_size, self.steps, self.eps))


class AdvVGG4(AdversarialModel):
    def __init__(self, input_shape, n_classes, width_scale=1, **kwargs):
        self.width_scale = width_scale

        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Conv2D(32 * width_scale, (5, 5), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64 * width_scale, (5, 5), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024 * width_scale, activation="relu")(x)
        outputs = tf.keras.layers.Dense(n_classes)(x)

        super(AdvVGG4, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

    @property
    def name(self):
        return f"VGG4_ws{self.width_scale}"


class VGG4(tf.keras.models.Model):
    def __init__(self, input_shape, n_classes, width_scale=1, **kwargs):
        self.width_scale = width_scale

        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Conv2D(32 * width_scale, (5, 5), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64 * width_scale, (5, 5), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024 * width_scale, activation="relu")(x)
        outputs = tf.keras.layers.Dense(n_classes)(x)

        super(VGG4, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

    @property
    def name(self):
        return f"VGG4_ws{self.width_scale}"


class VGG16(tf.keras.models.Model):
    def __init__(self, input_shape, n_classes, width_scale=1):
        self.width_scale = width_scale

        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Conv2D(64 * width_scale, (3, 3), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(128 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(256 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(256 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(256 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(512 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(512 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(512 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(strides=(2, 2))(x)
        x = tf.keras.layers.Conv2D(1024 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(1024 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(1024 * width_scale, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(strides=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096, activation='relu')(x)
        x = keras.layers.Dense(4096, activation='relu')(x)
        outputs = keras.layers.Dense(n_classes)(x)

        super(VGG16, self).__init__(inputs=inputs, outputs=outputs)

    @property
    def name(self):
        return f"VGG16_ws{self.width_scale}"


class MVGG4Builder(ModelBuilder):
    def __init__(self, n_filters=32, scale=1) -> None:
        self.n_filters = n_filters
        self.scale = scale

    def get_name(self):
        return 'MVGG4'

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


class SVGG4Builder(ModelBuilder):
    def __init__(self, n_filters=16) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'SVGG4'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model


class LVGG4Builder(ModelBuilder):
    def __init__(self, n_filters=64) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'LVGG4'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model


class LVGG4V2Builder(ModelBuilder):
    def __init__(self, n_filters=64) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'LVGG4V2'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (5, 5), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (5, 5), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            SplitFlatten(axis=-1),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model


class MVGG16Builder(ModelBuilder):
    def __init__(self, n_filters=64) -> None:
        self.n_filters = n_filters

    def get_name(self):
        return 'MVGG16'

    def build_model(self, input_shape, nb_classes):
        model = keras.models.Sequential([
            keras.layers.Conv2D(self.n_filters, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
            keras.layers.Conv2D(self.n_filters, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 2, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 2, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 4, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 4, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 4, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(self.n_filters * 8, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dense(nb_classes)
        ])
        return model