import base_frame as bf
import tensorflow as tf
import numpy as np
from dataset import DataSet, MNIST
from models import ModelBuilder, SVGG4Builder, MVGG4Builder, LVGG4Builder, LVGG4V2Builder
from activation_test import check_max_activations


def combine_weights(w_a, w_b, first_layer=False, last_layer=False, bias=False):
    """assert w_a.shape == w_b.shape

    if first_layer:
        return np.append(w_a, w_b, axis=-1)
    if last_layer != bias:
        return np.append(w_a, w_b, axis=0)
    if last_layer and bias:
        return np.add(w_a, w_b)

    w_shape = np.asarray(w_a.shape)
    w_shape[-1] *= 2
    w_shape[-2] *= 2

    return np.reshape(
        np.append(
            np.append(w_a, np.zeros_like(w_a), axis=0), np.append(np.zeros_like(w_b), w_b, axis=0),
            axis=1),
        w_shape)"""
    if first_layer:
        return np.append(w_a, w_b, axis=-1)
    if last_layer != bias:
        return np.append(w_a, w_b, axis=0)
    if last_layer and bias:
        #return np.mean([w_a, w_b], axis=0)
        return np.add(w_a, w_b)
    return np.block([[w_a, np.zeros_like(w_b)], [np.zeros_like(w_a), w_b]])


def combine_models(model_a: tf.keras.Model, model_b: tf.keras.Model, target_model: tf.keras.Model):
    new_weights = []

    n_layers = len(target_model.weights)
    new_weights.append(
        combine_weights(model_a.weights[0].numpy(), model_b.weights[0].numpy(), first_layer=True))
    for idx in range(1, n_layers - 2):
        new_weights.append(
            combine_weights(model_a.weights[idx].numpy(), model_b.weights[idx].numpy(), bias=idx % 2 != 0))
    new_weights.append(
        combine_weights(
            model_a.weights[n_layers - 2].numpy(),
            model_b.weights[n_layers - 2].numpy(),
            last_layer=True))
    new_weights.append(
        combine_weights(
            model_a.weights[n_layers - 1].numpy(),
            model_b.weights[n_layers - 1].numpy(),
            last_layer=True, bias=True))

    target_model.set_weights(new_weights)


def main():
    np.set_printoptions(threshold=np.inf)

    x_mnist = MNIST().get_train()[0]

    medium0 = MVGG4Builder().build_model((28, 28, 1), 10)
    medium1 = MVGG4Builder().build_model((28, 28, 1), 10)
    large = LVGG4V2Builder().build_model((28, 28, 1), 10)

    """combine_models(medium0, medium1, large)
    medium0_prediction = medium0.predict(x_mnist[0].reshape((1, 28, 28, 1)))
    medium1_prediction = medium1.predict(x_mnist[0].reshape((1, 28, 28, 1)))
    large_prediction = large.predict(x_mnist[0].reshape((1, 28, 28, 1)))
    print(medium0_prediction, np.argmax(medium0_prediction))
    print(medium1_prediction, np.argmax(medium1_prediction))
    print(large_prediction, np.argmax(large_prediction))"""

    combine_models(medium0, medium0, large)
    medium_activation = check_max_activations(medium0, x_mnist[0].reshape((1, 28, 28, 1)))
    large_activation = check_max_activations(large, x_mnist[0].reshape((1, 28, 28, 1)))
    with open("medium_activation.txt", "w") as fp:
        fp.write(str(medium_activation))
    with open("large_activation.txt", "w") as fp:
        fp.write(str(large_activation))

    """for idx in range(0, len(medium0.layers) + 1, 1):
        print(medium0.weights[idx].shape)
    print()
    for idx in range(0, len(large.layers) + 1, 1):
        print(large.weights[idx].shape)
    print()

    #print(medium0.weights[0].numpy)
    #print(medium1.weights[0].numpy)

    n_layers = len(large.weights)
    print(combine_weights(medium0.weights[0].numpy(), medium1.weights[0].numpy(), first_layer=True).shape)
    for idx in range(1, n_layers - 2):
        print(combine_weights(medium0.weights[idx].numpy(), medium1.weights[idx].numpy(), bias=idx % 2 != 0).shape)
    print(combine_weights(medium0.weights[n_layers - 2].numpy(), medium1.weights[n_layers - 2].numpy(), last_layer=False).shape)
    print(combine_weights(medium0.weights[n_layers - 1].numpy(), medium1.weights[n_layers - 1].numpy(), bias=True).shape)"""

    """w_a = np.random.random((5, 5, 1, 32))
    w_b = np.random.random((5, 5, 1, 32))
    #w_a = np.random.random((3136, 1024))
    #w_b = np.random.random((3136, 1024))

    print(combine_weights(w_a, w_b, first_layer=True).shape)"""


if __name__ == "__main__":
    bf.setup_base_frame(main)


"""
w_a = [1, 2] # 1x2
w_b = [3, 4] # 1x2
w = [[1, 0], [2, 0], [0, 3], [0, 4]] # 4x2
inp = [1, 2, 3, 4] # 1x4

w_a = np.array(w_a)
w_b = np.array(w_b)

np.stack((np.pad(w_a, (0, w_a.shape[0])), np.pad(w_b, (w_a.shape[0], 0))), axis=1)

np.append(np.append(a, np.zeros_like(a), axis=0), np.append(np.zeros_like(b), b, axis=0), axis=1)
np.stack((np.append(a, np.zeros_like(a), axis=0), np.append(np.zeros_like(b), b, axis=0)), axis=1)
np.reshape(np.stack((np.append(a, np.zeros_like(a), axis=0), np.append(np.zeros_like(b), b, axis=0)), axis=-2), np.add(a.shape, b.shape))
np.block([[a, np.zeros_like(b)], [np.zeros_like(a), b]])
"""