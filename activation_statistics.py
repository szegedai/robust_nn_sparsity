import base_frame as bf
import tensorflow as tf
import numpy as np
from dataset import MNIST
from activation_test import check_max_activations


def get_activation_statistics(activations: dict):
    total = 0
    zeros = 0
    for v in activations.values():
        total += np.size(v)
        zeros += np.count_nonzero(v == 0)
    return [total, zeros]


def main():
    x_mnist = MNIST().get_all()[0]

    robust100_50_model = tf.keras.models.load_model("robust_100_checkpoints/SampleCNN_050.h5")
    natural50_model = tf.keras.models.load_model("natural_50_checkpoints/SampleCNN_050.h5")

    robust100_50_activations = check_max_activations(robust100_50_model, x_mnist)
    natural50_activations = check_max_activations(natural50_model, x_mnist)

    print("robust (50 epochs) model neuron activations:")
    print("\ttotal: {}\n\tzeros: {}".format(*get_activation_statistics(robust100_50_activations)))

    print("natural (50 epochs) model neuron activations:")
    print("\ttotal: {}\n\tzeros: {}".format(*get_activation_statistics(natural50_activations)))


if __name__ == "__main__":
    bf.setup_base_frame(main)
