import base_frame as bf
import tensorflow as tf
import numpy as np
import time
from dataset import MNIST
from activation_test import read_activation_file


#np.set_printoptions(threshold=np.inf)


def get_layer_index(model, layer_name):
    for i in range(len(model.layers)):
        if model.layers[i] == model.get_layer(layer_name):
            return i
    return None

# Note: WIP!
def prune_model(original_model: tf.keras.Model, activations):
    pass


# Zeroes out the input weights of inactive neurons
# Note: This is an older implementation
"""
def soft_prune_model(original_model: tf.keras.Model, activations):
    new_model = tf.keras.models.clone_model(original_model)
    new_model.set_weights(original_model.get_weights())

    for layer_name, layer_activations in activations.items():
        layer_ref = new_model.get_layer(layer_name)
        inactive_indices = []

        #inactive_indices = np.argwhere(activations[layer_name] == 0)

        for i in range(activations[layer_name].shape[-1]):
            if np.all(activations[layer_name][..., i] == 0):
                inactive_indices.append(i)

        weights, biases = layer_ref.get_weights()
        for i in inactive_indices:
            weights[..., i] = 0
            biases[i] = 0                
        layer_ref.set_weights([weights, biases])
    return new_model
"""


# Zeroes out the input weights of inactive neurons and also the output weights
def soft_prune_model(original_model: tf.keras.Model, activations: dict, threshold=0.0):
    new_model = tf.keras.models.clone_model(original_model)
    new_model.set_weights(original_model.get_weights())

    prev_inactive_indices = None
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name in activations:
            layer_activations = activations[layer_name]
            inactive_indices = []

            weights, biases = layer.get_weights()
            if prev_inactive_indices is not None:
                for i in prev_inactive_indices:
                    weights[..., i, :] = 0

            for i in range(layer_activations.shape[-1]):
                if np.all(layer_activations[..., i] <= threshold):
                    inactive_indices.append(i)
            prev_inactive_indices = inactive_indices.copy()

            for i in inactive_indices:
                weights[..., i] = 0
                biases[i] = 0
            layer.set_weights([weights, biases])

    return new_model


def soft_prune_model2(original_model: tf.keras.Model, activations: dict, threshold=0.0):
    prev_inactive_indices = None
    for layer in original_model.layers:
        layer_name = layer.name
        if layer_name in activations:
            layer_activations = activations[layer_name]
            inactive_indices = []

            weights, biases = layer.get_weights()
            if prev_inactive_indices is not None:
                for i in prev_inactive_indices:
                    weights[..., i, :] = 0

            for i in range(layer_activations.shape[-1]):
                if np.all(layer_activations[..., i] <= threshold):
                    inactive_indices.append(i)
            prev_inactive_indices = inactive_indices.copy()

            for i in inactive_indices:
                weights[..., i] = 0
                biases[i] = 0
            layer.set_weights([weights, biases])


def main():
    # Check for inactive convolutional filters:
    """
    for i in range(activations["conv2d"].shape[-1]):
        if np.all(activations["conv2d"][:, :, i] == 0):
            print("[conv2d] inactive kernel index:", i)
    for i in range(activations["conv2d_1"].shape[-1]):
        if np.all(activations["conv2d_1"][:, :, i] == 0):
            print("[conv2d_1] inactive kernel index:", i)
    quit()
    """


if __name__ == "__main__":
    bf.setup_base_frame(main)
