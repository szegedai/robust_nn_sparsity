import base_frame as bf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from foolbox.attacks import LinfPGD
from foolbox.models import TensorFlowModel
from dataset import MNIST
from activation_test import read_activation_file, check_max_activations
from model_pruning import soft_prune_model
from train import batch_attack

# Reminder: model.get_weights() returns a list where every odd index represents the biases.


def check_model_inequality(model_a, model_b, dataset_x):
    pred_a = model_a.predict(dataset_x)
    pred_b = model_b.predict(dataset_x)
    return np.mean(np.abs(pred_a - pred_b))


def get_accuracy(model, dataset):
    return model.evaluate(dataset[0], dataset[1], verbose=0)[1]


def get_adversarial_accuracy(model, dataset, batch_size=50, step_size=0.01, steps=40, epsilons=0.3):
    dataset = tf.constant(dataset[0]), tf.constant(dataset[1])
    attack = LinfPGD(abs_stepsize=step_size, steps=steps, random_start=False)
    f_model = TensorFlowModel(model, bounds=(0, 1), device="/device:GPU:0")

    #x_adv, _, _ = attack(f_model, dataset[0], dataset[1], epsilons=epsilons)
    x_adv = batch_attack(dataset[0], dataset[1], attack, f_model, epsilons, batch_size)

    return model.evaluate(x_adv, dataset[1], verbose=0)[1]


def print_parameter_infos(original_model, new_model, include_biases=True, header=None):
    step_size = 1 if include_biases else 2
    print()
    if header is not None:
        print(header)
    parameter_count = 0
    for i in original_model.get_weights()[::step_size]:
        parameter_count += np.prod(i.shape)
    print("architecture parameter count:", parameter_count)

    zero_count = 0
    for i in original_model.get_weights()[::step_size]:
        zero_count += np.count_nonzero(i == 0)
    print("original count of zeros:", zero_count)
    print(f"original percentage of zeros: {zero_count / parameter_count * 100}%")

    zero_count = 0
    for i in new_model.get_weights()[::step_size]:
        zero_count += np.count_nonzero(i == 0)
    print("new count of zeros:", zero_count)
    print(f"new percentage of zeros: {zero_count / parameter_count * 100}%")
    print(f"(biases {'not ' if not include_biases else ''}included)")
    print()


def get_weight_histogram(model: tf.keras.Model, bins=100, ranges=None, plot=True, plot_title=None):
    weights = []
    for w in model.get_weights()[::2]:
        weights += w.ravel().tolist()
    weights = np.array(weights)
    if ranges is None:
        ranges = (np.min(weights), np.max(weights))
    counts, bins = np.histogram(weights, bins=bins, range=ranges)
    if plot:
        if plot_title is not None:
            plt.title(plot_title)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
    return counts, bins


def get_activation_histogram(activations: dict, bins=100, ranges=None, plot=True, plot_title=None):
    flat_activations = []
    for v in activations.values():
        flat_activations += v.ravel().tolist()
    flat_activations = np.array(flat_activations)
    if ranges is None:
        ranges = (np.min(flat_activations), np.max(flat_activations))
    counts, bins = np.histogram(flat_activations, bins=bins, range=ranges)
    if plot:
        if plot_title is not None:
            plt.title(plot_title)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()
    return counts, bins


def main():
    """mnist = MNIST(seed=9).get_val()

    robust_activations = read_activation_file("robust_activations.json")
    natural_activations = read_activation_file("natural_activations.json")

    robust_original_model = tf.keras.models.load_model("saved_models/robust_50.h5")
    natural_original_model = tf.keras.models.load_model("saved_models/natural.h5")

    robust_pruned_model = soft_prune_model(robust_original_model, robust_activations, threshold=0.0)
    natural_pruned_model = soft_prune_model(natural_original_model, natural_activations, threshold=0.0)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']

    robust_original_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    robust_pruned_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    natural_original_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    natural_pruned_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)

    # parameters of models:
    print_parameter_infos(robust_original_model, robust_pruned_model, include_biases=False, header="robust model")
    print_parameter_infos(natural_original_model, natural_pruned_model, include_biases=False, header="natural model")

    # model inequalities:
    print("inequality of robust models:",
          check_model_inequality(robust_original_model, robust_pruned_model, mnist[0]))
    print("inequality of natural models:",
          check_model_inequality(natural_original_model, natural_pruned_model, mnist[0]))

    # accuracy changes:
    print("robust original model acc, adv_acc:",
          get_accuracy(robust_original_model, mnist),
          get_adversarial_accuracy(robust_original_model, mnist))
    print("robust pruned model acc, adv_acc:",
          get_accuracy(robust_pruned_model, mnist),
          get_adversarial_accuracy(robust_pruned_model, mnist))
    print("natural original model acc, adv_acc:",
          get_accuracy(natural_original_model, mnist),
          get_adversarial_accuracy(natural_original_model, mnist))
    print("natural pruned model acc, adv_acc:",
          get_accuracy(natural_pruned_model, mnist),
          get_adversarial_accuracy(natural_pruned_model, mnist))

    # weight histograms:
    get_weight_histogram(robust_original_model, plot_title="robust original weights")
    get_weight_histogram(robust_pruned_model, plot_title="robust pruned weights")
    get_weight_histogram(natural_original_model, plot_title="natural original weights")
    get_weight_histogram(natural_pruned_model, plot_title="natural pruned weights")

    # activation histograms
    get_activation_histogram(robust_activations, plot_title="robust activations")
    get_activation_histogram(natural_activations, plot_title="natural activations")"""


    """new_robust_model = tf.keras.models.load_model("robust_50_pruned_checkpoints/SampleCNN_050-0.20.h5")
    print("adv_acc:",
          get_adversarial_accuracy(new_robust_model, mnist))

    robust_original_model = tf.keras.models.load_model("saved_models/robust_50.h5")
    print("robust original model adv_acc:",
          get_adversarial_accuracy(robust_original_model, mnist))"""

    #model = tf.keras.models.load_model("saved_models/robust_100.h5")
    model = tf.keras.models.load_model("saved_models/robust_50.h5")
    pruned = soft_prune_model(model, check_max_activations(model, MNIST().get_val()[0]))

    print_parameter_infos(model, pruned)


if __name__ == "__main__":
    bf.setup_base_frame(main)
