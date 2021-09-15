import base_frame as bf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from dataset import DataSet, MNIST
from models import SplitFlatten, ModelBuilder, SVGG4Builder, MVGG4Builder, LVGG4Builder, LVGG4V2Builder
from train import train_hybrid, train_combined
from activation_test import check_max_activations
from activation_statistics import get_activation_statistics


def run_hybrid_training_experiment(model_builder: ModelBuilder, dataset: DataSet, main_save_dir, split, idx=0):
    checkpoint_save_dir = f"hybrid_{split[0]}_{split[1]}_checkpoints"

    x_test, y_test = dataset.get_test()
    x_all, y_all = dataset.get_all()

    epochs = split[0] + split[1]
    epoch_indices = np.arange(1, epochs + 1, 1)

    train_hybrid(model_builder, dataset, nat_epochs=split[0], adv_epochs=split[1],
                 save_dir=f"{main_save_dir}/{checkpoint_save_dir}{idx}")

    activation_rates = []
    accuracies = []
    for i in epoch_indices:
        print(f"{i}/{epochs}", end="\r")
        hybrid_model = tf.keras.models.load_model(
            glob(f"{main_save_dir}/{checkpoint_save_dir}{idx}/*{i:03d}*.h5")[0])

        total, inactive = get_activation_statistics(check_max_activations(hybrid_model, x_all))
        activation_rates.append(inactive / total)

        loss, acc = hybrid_model.evaluate(x_test, y_test)
        accuracies.append(acc)

    plt.plot(epoch_indices, activation_rates, color="red")
    plt.savefig(f"{main_save_dir}/hybrid_activation{idx}.png")
    plt.clf()

    plt.plot(epoch_indices, accuracies, color="blue")
    plt.savefig(f"{main_save_dir}/hybrid_test_accuracy{idx}.png")
    plt.clf()


def run_combined_training_experiment(initial_model_builder: ModelBuilder, result_model_builder: ModelBuilder, dataset: DataSet, main_save_dir, split, idx=0):
    checkpoint_save_dir = f"combined_{split[0]}_{split[1]}_checkpoints"

    x_test, y_test = dataset.get_test()
    x_all, y_all = dataset.get_all()

    epochs = split[0] + split[1]
    epoch_indices = np.arange(1, epochs + 1, 1)

    train_combined(initial_model_builder, result_model_builder, dataset, nat_epochs=split[0], adv_epochs=split[1],
                   save_dir=f"{main_save_dir}/{checkpoint_save_dir}{idx}")

    activation_rates = []
    accuracies = []
    for i in epoch_indices:
        print(f"{i}/{epochs}", end="\r")
        hybrid_model = tf.keras.models.load_model(
            glob(f"{main_save_dir}/{checkpoint_save_dir}{idx}/*{i:03d}*.h5")[0],
            custom_objects={"SplitFlatten": SplitFlatten})

        total, inactive = get_activation_statistics(check_max_activations(hybrid_model, x_all))
        activation_rates.append(inactive / total)

        loss, acc = hybrid_model.evaluate(x_test, y_test)
        accuracies.append(acc)

    plt.plot(epoch_indices, activation_rates, color="red")
    plt.plot(epoch_indices, accuracies, color="blue")
    plt.savefig(f"{main_save_dir}/combined_test_accuracy_activation{idx}.png")
    plt.clf()


def main():
    main_save_dir = "hybrid_training"
    sample_size = 1
    mnist = MNIST()
    """for i in range(sample_size):
        run_hybrid_training_experiment(SVGG4Builder(), mnist, f"{main_save_dir}/small", (5, 45), i)
    print("")
    for i in range(sample_size):
        run_hybrid_training_experiment(MVGG4Builder(), mnist, f"{main_save_dir}/medium", (5, 45), i)
    print("")
    for i in range(sample_size):
        run_hybrid_training_experiment(LVGG4Builder(), mnist, f"{main_save_dir}/large", (5, 45), i)"""

    """main_save_dir = "combined_training"
    sample_size = 10
    for i in range(sample_size):
        run_combined_training_experiment(MVGG4Builder(), LVGG4Builder(), mnist, f"{main_save_dir}/MtoL", (5, 45), i)
    for i in range(sample_size):
        run_combined_training_experiment(MVGG4Builder(), LVGG4V2Builder(), mnist, f"{main_save_dir}/MtoLv2", (5, 45), i)"""

    """model = tf.keras.models.load_model(
        "remote/combined_training/MtoL/combined_5_45_checkpoints0/2xMVGG4=LVGG4_005-14.31.h5")
    from combined_training import combine_models
    large: tf.keras.models.Model = LVGG4Builder().build_model((28, 28, 1), 10)
    large.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    combine_models(model, model, large)
    x_mnist, y_mnist = mnist.get_test()
    print(large.evaluate(x=x_mnist, y=y_mnist)) # ==> 11.39% acc"""

    np.set_printoptions(threshold=np.inf)
    model = tf.keras.models.load_model("remote/combined_training/MtoL/combined_5_45_checkpoints0/2xMVGG4=LVGG4_006-1.66.h5")
    with open("tmp.txt", "w") as fp:
        fp.write(str(model.get_weights()))


if __name__ == "__main__":
    bf.setup_base_frame(main)
