import base_frame as bf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from dataset import MNIST
from activation_test import check_max_activations
from activation_statistics import get_activation_statistics


def main():
    epochs = 50
    x_mnist = MNIST().get_all()

    epoch_indices = np.arange(1, epochs + 1, 1)
    natural_activation_percentages = []
    robust_activation_percentages = []
    for i in epoch_indices:
        natural_model = tf.keras.models.load_model(glob("natural_50_checkpoints/SampleCNN_{:03d}*.h5".format(i))[0])
        robust_model = tf.keras.models.load_model(glob("robust_100_checkpoints/SampleCNN_{:03d}*.h5".format(i))[0])

        total, inactive = get_activation_statistics(check_max_activations(natural_model, x_mnist))
        natural_activation_percentages.append(inactive / total)
        total, inactive = get_activation_statistics(check_max_activations(robust_model, x_mnist))
        robust_activation_percentages.append(inactive / total)

        print(f"{i}/{epochs}", end="\r")

    plt.plot(epoch_indices, natural_activation_percentages, label="natural", color="blue")
    plt.plot(epoch_indices, robust_activation_percentages, label="robust", color="red")
    plt.savefig("activation_comparisons.png")
    plt.show()


if __name__ == "__main__":
    bf.setup_base_frame(main)
