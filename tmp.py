import base_frame as bf
import tensorflow as tf
import numpy as np
from models import AdvVGG4, VGG4, PGD_attack
from dataset import MNIST
from train import train_nat, train_adv
from activation_test import check_max_activations
from activation_statistics import get_activation_statistics


def main():
    mnist = MNIST()

    adv_model = AdvVGG4((28, 28, 1), 10, step_size=0.01, steps=40, eps=0.3)
    adv_model.compile(tf.keras.optimizers.Adam(),
                      tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["sparse_categorical_accuracy"])
    """adv_model.fit(*mnist.get_train(), batch_size=50, epochs=5)
    adv_model.evaluate(*mnist.get_test())"""

    #print(get_activation_statistics(check_max_activations(adv_model, mnist.get_all()[0])))

    import matplotlib.pyplot as plt
    from foolbox.attacks import LinfPGD
    from foolbox.models import TensorFlowModel
    attack = LinfPGD(abs_stepsize=0.3, steps=1, random_start=False)
    fmodel = TensorFlowModel(adv_model, bounds=(0, 1), device='/device:GPU:0')
    x, y = mnist.get_train()
    x = x[0:1]
    y = y[0:1]
    image = PGD_attack(adv_model, x, y, step_size=0.3, steps=1, eps=0.3, random_start=False)[0].numpy()
    image2 = PGD_attack(adv_model, x, y, step_size=0.3, steps=1, eps=0.3, random_start=False)[0].numpy()
    fb_image = attack(fmodel, tf.identity(x), tf.identity(y), epsilons=0.3)[0].numpy()
    fb_image2 = attack(fmodel, tf.identity(x), tf.identity(y), epsilons=0.3)[0].numpy()

    fig, ax = plt.subplots(1, 3)

    """ax[0].imshow(fb_image2[0], cmap='gray')
    ax[1].imshow(fb_image[0], cmap='gray')
    ax[2].imshow(np.abs(fb_image[0] - fb_image2[0]), cmap='gray')"""

    ax[0].imshow(image2[0], cmap='gray')
    ax[1].imshow(image[0], cmap='gray')
    ax[2].imshow(np.abs(image[0] - image2[0]), cmap='gray')

    plt.show()


if __name__ == "__main__":
    bf.setup_base_frame(main)
