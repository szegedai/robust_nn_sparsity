import base_frame as bf
import tensorflow as tf
import numpy as np
from dataset import MNIST
import time
import json

# Just for testing!
#np.set_printoptions(threshold=np.inf)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_activation_file(activation_file):
    activation_data = None
    with open(activation_file, "r") as fp:
        activation_data = json.load(fp)
    for k, v in activation_data.items():
        activation_data[k] = np.array(v)
    return activation_data


def check_max_activations(model: tf.keras.Model, x_data_set, batch_size=5120):
    activations = dict()
    extractors = dict()
    batches = tf.data.Dataset.from_tensor_slices(x_data_set).batch(batch_size)
    for layer in model.layers:
        if layer.count_params() > 0:
            extractors[layer.name] = tf.keras.Model(model.inputs, layer.output)
            activations[layer.name] = np.zeros(layer.output.shape[1:], dtype=np.float)
            for b in batches:
                batch_max_activations = np.max(extractors[layer.name].predict(b), axis=0)
                activations[layer.name] = np.maximum(activations[layer.name], batch_max_activations)
    return activations


def main():
    model = tf.keras.models.load_model(bf.arg_flags.model_file)
    x_mnist = MNIST(seed=1).get_all()[0]

    res = check_max_activations(model, x_mnist, bf.arg_flags.batch_size)

    with open(bf.arg_flags.result_file, "w") as fp:
        json.dump(res, fp, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    bf.arg_parser.add_argument("--batch_size", type=int, default=5120)
    bf.arg_parser.add_argument("--result_file", type=str, default="activations.json")
    bf.arg_parser.add_argument("--model_file", type=str, required=True)

    bf.setup_base_frame(main)
