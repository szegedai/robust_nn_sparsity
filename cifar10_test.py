import base_frame as bf
import tensorflow as tf
import numpy as np
from models import VGG4, VGG16
from dataset import CIFAR10, MNIST
from train import train_nat, train_adv
from activation_test import check_max_activations
from activation_statistics import get_activation_statistics


def main():
    cifar10 = CIFAR10()

    adv_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), classes=10, pooling="max", weights=None)
    train_adv(adv_model, cifar10, epochs=155, eps=8/255, step_size=2/255, steps=10)

    nat_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), classes=10, pooling="max", weights=None)
    train_nat(nat_model, cifar10, epochs=155)

    with open("cifar10_resnet50_neuron_inactivity.txt", "w") as fp:
        fp.write(str(get_activation_statistics(check_max_activations(adv_model, cifar10.get_all()[0]))))
        fp.write(str(get_activation_statistics(check_max_activations(nat_model, cifar10.get_all()[0]))))

    """"
    mnist = MNIST()
    
    #adv_model = VGG16((32, 32, 3), 10)
    adv_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), classes=10, pooling="max", weights=None)
    train_adv(adv_model, mnist, 1)

    #nat_model = VGG16((32, 32, 3), 10)
    nat_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), classes=10, pooling="max", weights=None)
    train_nat(nat_model, mnist, 1)

    print(get_activation_statistics(check_max_activations(adv_model, mnist.get_all()[0])))
    print(get_activation_statistics(check_max_activations(nat_model, mnist.get_all()[0])))"""


if __name__ == "__main__":
    bf.setup_base_frame(main)
