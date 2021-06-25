import sys
import tensorflow as tf
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_flags = None


def setup_base_frame(main_fn):
    if list(sys._current_frames().values())[0].f_back.f_globals['__name__'] != "__main__":
        return
    global arg_parser
    global arg_flags

    arg_parser.add_argument("--gpu", type=int, default=0)
    arg_parser.add_argument("--memory_limit", type=int, default=1024)

    arg_flags = arg_parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    selected = gpus[arg_flags.gpu]
    tf.config.experimental.set_visible_devices(selected, "GPU")
    tf.config.experimental.set_memory_growth(selected, True)
    tf.config.experimental.set_virtual_device_configuration(
        selected,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=arg_flags.memory_limit)])
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    l_gpu = logical_gpus[0]
    with tf.device(l_gpu.name):
        main_fn()
