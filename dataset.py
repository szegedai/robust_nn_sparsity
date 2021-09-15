from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class DataSet(ABC):
    @abstractmethod
    def get_train(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_val(self):
        pass

    @abstractmethod
    def get_all(self):
        pass


class MNIST(DataSet):
    def __init__(self, val_size=1000, seed=9) -> None:
        self.rnd = np.random.RandomState(seed)
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32)
        self.x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        self.y_train = np.array(y_train, np.int64)
        self.y_test = np.array(y_test, np.int64)
        self.x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train,
                                                                             self.y_train)

    def split_data(self, rnd, sample_per_class, x, y):
        x_equalized = ()
        x_remained = ()
        y_equalized = ()
        y_remained = ()
        for i in np.unique(y):
            idxs = rnd.permutation(np.sum(y == i))
            x_i = x[y == i]
            y_i = y[y == i]
            x_equalized = x_equalized + (x_i[idxs[:sample_per_class]],)
            y_equalized = y_equalized + (y_i[idxs[:sample_per_class]],)
            x_remained = x_remained + (x_i[idxs[sample_per_class:]],)
            y_remained = y_remained + (y_i[idxs[sample_per_class:]],)
        return np.concatenate(x_remained, axis=0), np.concatenate(y_remained, axis=0), \
               np.concatenate(x_equalized, axis=0), np.concatenate(y_equalized, axis=0)

    def get_bound(self):
        return (0., 1.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0]

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_all(self):
        return np.concatenate((self.x_train, self.x_test, self.x_val)), \
               np.concatenate((self.y_train, self.y_test, self.y_val))

    def get_name(self):
        return 'MNIST'


class CIFAR10(DataSet):
    def __init__(self, val_size=1000, seed=9) -> None:
        self.rnd = np.random.RandomState(seed)
        cifar10 = tf.keras.datasets.cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = np.array(x_train / 255.0, np.float32), np.array(x_test / 255.0, np.float32)
        self.x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
        self.y_train = np.array(y_train, np.int64).reshape(-1)
        self.y_test = np.array(y_test, np.int64).reshape(-1)
        self.x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(self.rnd, val_size // 10, self.x_train,
                                                                             self.y_train)

    def split_data(self, rnd, sample_per_class, x, y):
        x_equalized = ()
        x_remained = ()
        y_equalized = ()
        y_remained = ()
        for i in np.unique(y):
            idxs = rnd.permutation(np.sum(y == i))
            x_i = x[y == i]
            y_i = y[y == i]
            x_equalized = x_equalized + (x_i[idxs[:sample_per_class]],)
            y_equalized = y_equalized + (y_i[idxs[:sample_per_class]],)
            x_remained = x_remained + (x_i[idxs[sample_per_class:]],)
            y_remained = y_remained + (y_i[idxs[sample_per_class:]],)
        return np.concatenate(x_remained, axis=0), np.concatenate(y_remained, axis=0), \
               np.concatenate(x_equalized, axis=0), np.concatenate(y_equalized, axis=0)

    def get_bound(self):
        return (0., 1.)

    def get_input_shape(self):
        return self.x_train.shape[1:]

    def get_nb_classes(self):
        return np.unique(self.y_train).shape[0]

    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test

    def get_val(self):
        return self.x_val, self.y_val

    def get_all(self):
        return np.concatenate((self.x_train, self.x_test, self.x_val)), \
               np.concatenate((self.y_train, self.y_test, self.y_val))

    def get_name(self):
        return 'CIFAR10'
