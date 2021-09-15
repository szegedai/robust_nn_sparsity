import base_frame as bf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from dataset import MNIST
from models import MVGG4Builder, SplitFlatten
from train import train_pert
from activation_test import check_max_activations


def get_activation_statistics(activations: dict):
    total = 0
    zeros = 0
    for v in activations.values():
        total += np.size(v)
        zeros += np.count_nonzero(v == 0)
    return [total, zeros]


def plot_detailed_activation_statistics(dir_location, output_file, x_dataset, epoch_count, starter_epoch=0, fig_size=(24, 5), verbose=False):
    all_activations = {key: [] for key in check_max_activations(
        tf.keras.models.load_model(
            glob(f"{dir_location}/*_{starter_epoch + 1:03d}*.h5")[0],
            custom_objects={"SplitFlatten": SplitFlatten}),
        x_dataset).keys()}
    layer_neuron_counts = dict.fromkeys(all_activations.keys(), -1)
    for i in range(starter_epoch, epoch_count + starter_epoch):
        model = tf.keras.models.load_model(
            glob(f"{dir_location}/*_{i + 1:03d}*.h5")[0],
            custom_objects={"SplitFlatten": SplitFlatten})
        activations = check_max_activations(model, x_dataset)
        for key, value in activations.items():
            if layer_neuron_counts[key] == -1:
                layer_neuron_counts[key] = np.size(value)
            zeros = np.count_nonzero(value == 0)
            all_activations[key].append(zeros)
        if verbose:
            print(f"{i + 1}/{epoch_count + starter_epoch}", end="\r")
    plt.figure(1)
    plt.subplots(figsize=fig_size)
    i = 1
    for key, value in all_activations.items():
        plt.subplot(140 + i)
        plt.plot(range(epoch_count), value, color="red")
        plt.axis([0, epoch_count - 1, 0, layer_neuron_counts[key]])
        plt.ylabel(f"{key} inactive neurons")
        i += 1
    plt.savefig(output_file)
    plt.clf()

def main():
    x_mnist = MNIST().get_all()[0]

    """robust100_50_model = tf.keras.models.load_model("robust_100_checkpoints/SampleCNN_050.h5")
    natural50_model = tf.keras.models.load_model("natural_50_checkpoints/SampleCNN_050.h5")

    robust100_50_activations = check_max_activations(robust100_50_model, x_mnist)
    natural50_activations = check_max_activations(natural50_model, x_mnist)

    print("robust (50 epochs) model neuron activations:")
    print("\ttotal: {}\n\tzeros: {}".format(*get_activation_statistics(robust100_50_activations)))

    print("natural (50 epochs) model neuron activations:")
    print("\ttotal: {}\n\tzeros: {}".format(*get_activation_statistics(natural50_activations)))"""

    """model = train_pert(MVGG4Builder(), MNIST(), epochs=50, batch_size=32, max_perturbation=0.5)
    model_activations = check_max_activations(model, x_mnist)
    print("\ttotal: {}\n\tzeros: {}".format(*get_activation_statistics(model_activations)))
    # 0.5:
    # 2189 - 50 epochs
    # 1505 - 10 epochs
    # 1099 - 5 epochs
    # 367 - 1 epochs
    # 0.1:
    # 672 - 50 epochs
    # 581 - 10 epochs
    # 429 - 5 epochs
    # 311 - 1 epoch
    # 0.01:
    # 458 - 50 epochs
    # 338 - 10 epochs
    # 249 - 5 epochs
    # 244 - 1 epoch"""

    for i in range(5):
        plot_detailed_activation_statistics(
            f"remote/combined_training/MtoLv2/combined_5_45_checkpoints{i}",
            f"hybridv2_5_45_activations{i}_0-5.png",
            x_mnist, 5, 0)
        plot_detailed_activation_statistics(
            f"remote/combined_training/MtoLv2/combined_5_45_checkpoints{i}",
            f"hybridv2_5_45_activations{i}_6-50.png",
            x_mnist, 45, 5)
        plot_detailed_activation_statistics(
            f"remote/combined_training/MtoL/combined_5_45_checkpoints{i}",
            f"hybrid_5_45_activations{i}_0-5.png",
            x_mnist, 5, 0)
        plot_detailed_activation_statistics(
            f"remote/combined_training/MtoL/combined_5_45_checkpoints{i}",
            f"hybrid_5_45_activations{i}_6-50.png",
            x_mnist, 45, 5)


if __name__ == "__main__":
    bf.setup_base_frame(main)
