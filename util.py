import os


def mk_parent_dir(f_name):
    if not os.path.isdir(os.path.dirname(f_name)):
        os.makedirs(os.path.dirname(f_name))