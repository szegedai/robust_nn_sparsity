import base_frame as bf
import os
import tensorflow as tf
import numpy as np
from dataset import MNIST
from models import MVGG4Builder
import util
import time
from foolbox.attacks import LinfPGD
from foolbox.models import TensorFlowModel
from activation_test import check_max_activations
from model_pruning import soft_prune_model2
from combined_training import combine_models


def batch_attack(imgs, labels, attack, fmodel, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(fmodel, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)


"""def train_adv(model_holder, dataset, epochs=50, batch_size=32, pruning_fn=None, seed=9):
    np.random.seed(seed)
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model = model_holder.build_model(dataset.get_input_shape(), dataset.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    m_path = os.path.join(bf.arg_flags.save_dir, model_holder.get_name())
    util.mk_parent_dir(m_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                 tf.keras.callbacks.CSVLogger(os.path.join(bf.arg_flags.save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    attack = LinfPGD(abs_stepsize=bf.arg_flags.step_size, steps=bf.arg_flags.steps, random_start=True)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': bf.arg_flags.batch_size, 'epochs': epochs,
             'steps': x_train.shape[0] // bf.arg_flags.batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        # soft pruning of the model by activations:
        if pruning_fn is not None:
            activations = check_max_activations(model, dataset.get_all(), batch_size=2048)
            pruning_fn(model, activations, threshold=0.0)
        # --------------------------------------- #
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_adv_batch, _, success = attack(fmodel, x_batch,
                                             criterion=y_batch, epsilons=bf.arg_flags.eps)
            batch_eval = model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, bf.arg_flags.eps, bf.arg_flags.batch_size)
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        val_eval = model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break
    return model"""


def train_adv(model: tf.keras.models.Model, dataset, epochs=50, batch_size=32, eps=0.3, step_size=0.01, steps=40, save_dir=None, pruning_fn=None, seed=9):
    np.random.seed(seed)
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    callbacks = []
    if save_dir is not None:
        m_path = os.path.join(save_dir, model.name)
        util.mk_parent_dir(m_path)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                     tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    attack = LinfPGD(abs_stepsize=step_size, steps=steps, random_start=True)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': epochs,
             'steps': x_train.shape[0] // batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        # soft pruning of the model by activations:
        if pruning_fn is not None:
            activations = check_max_activations(model, dataset.get_all(), batch_size=2048)
            pruning_fn(model, activations, threshold=0.0)
        # --------------------------------------- #
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_adv_batch, _, success = attack(fmodel, x_batch,
                                             criterion=y_batch, epsilons=eps)
            batch_eval = model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, eps, batch_size)
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        val_eval = model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break
    return model


def train_nat(model: tf.keras.models.Model, dataset, epochs=50, batch_size=32, save_dir=None, pruning_fn=None, seed=9):
    np.random.seed(seed)
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    callbacks = []
    if save_dir is not None:
        m_path = os.path.join(save_dir, model.name)
        util.mk_parent_dir(m_path)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}.h5'),
                     tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
    return model


def train_pert(model_holder, dataset, max_perturbation=0.1, epochs=50, batch_size=32, save_dir=None, seed=9):
    np.random.seed(seed)
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model = model_holder.build_model(dataset.get_input_shape(), dataset.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    callbacks = []
    if save_dir is not None:
        m_path = os.path.join(save_dir, model_holder.get_name())
        util.mk_parent_dir(m_path)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                     tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': epochs,
             'steps': x_train.shape[0] // batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_pert_batch = x_batch + np.random.uniform(0.0, max_perturbation, x_batch.shape)
            batch_eval = model.train_on_batch(x_pert_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        val_eval = model.evaluate(x_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break
    return model


def train_hybrid(model_holder, dataset, nat_epochs=10, adv_epochs=40, batch_size=32, eps=0.3, step_size=0.01, steps=40, save_dir=None, seed=9):
    np.random.seed(seed)
    epochs = nat_epochs + adv_epochs
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    model = model_holder.build_model(dataset.get_input_shape(), dataset.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    callbacks = []
    if save_dir is not None:
        m_path = os.path.join(save_dir, model_holder.get_name())
        util.mk_parent_dir(m_path)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                     tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    attack = LinfPGD(abs_stepsize=step_size, steps=steps, random_start=True)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': epochs,
             'steps': x_train.shape[0] // batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            if i < nat_epochs:
                batch_eval = model.train_on_batch(x_batch, y_batch)
            else:
                x_adv_batch, _, success = attack(fmodel, x_batch, criterion=y_batch, epsilons=eps)
                batch_eval = model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, eps, batch_size)
        val_eval = model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        if model.stop_training:
            break
    return model


def train_combined(initial_model_holder, result_model_holder, dataset, nat_epochs=10, adv_epochs=40, batch_size=32, eps=0.3, step_size=0.01, steps=40, save_dir=None, seed=9):
    np.random.seed(seed)
    epochs = nat_epochs + adv_epochs
    x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_val()
    print(x_train.shape, np.bincount(y_train), x_val.shape, np.bincount(y_val))
    initial_model = initial_model_holder.build_model(dataset.get_input_shape(), dataset.get_nb_classes())
    result_model = result_model_holder.build_model(dataset.get_input_shape(), dataset.get_nb_classes())
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['sparse_categorical_accuracy']
    initial_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    result_model.compile(tf.keras.optimizers.Adam(1e-4), loss_fn, metrics)
    callbacks = []
    if save_dir is not None:
        m_path = os.path.join(save_dir, f"2x{initial_model_holder.get_name()}={result_model_holder.get_name()}")
        util.mk_parent_dir(m_path)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(m_path + '_{epoch:03d}-{val_loss:.2f}.h5'),
                     tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'))]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(seed=seed, buffer_size=x_train.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)

    attack = LinfPGD(abs_stepsize=step_size, steps=steps, random_start=True)
    fmodel = TensorFlowModel(initial_model, bounds=(0, 1), device='/device:GPU:0')
    imgs_val, labels_val = tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)
    for cb in callbacks:
        cb.set_model(initial_model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': nat_epochs,
             'steps': x_train.shape[0] // batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    for i in range(nat_epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            batch_eval = initial_model.train_on_batch(x_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, eps, batch_size)
        val_eval = initial_model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (nat_epochs - 1) or initial_model.stop_training:
                cb.on_train_end()
        if initial_model.stop_training:
            break
    combine_models(initial_model, initial_model, result_model)
    for cb in callbacks:
        cb.set_model(result_model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': adv_epochs,
             'steps': x_train.shape[0] // batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    fmodel = TensorFlowModel(result_model, bounds=(0, 1), device='/device:GPU:0')
    for i in range(adv_epochs):
        for cb in callbacks:
            cb.on_epoch_begin(i + nat_epochs)
        delta0 = time.time()
        a_loss, a_acc = 0, 0
        for b_idx, (x_batch, y_batch) in enumerate(train_dataset):
            print('\r', b_idx, end=' ')
            x_adv_batch, _, success = attack(fmodel, x_batch, criterion=y_batch, epsilons=eps)
            batch_eval = result_model.train_on_batch(x_adv_batch, y_batch)
            a_loss = a_loss + batch_eval[0]
            a_acc = a_acc + batch_eval[1]
        train_eval = [a_loss / (b_idx + 1), a_acc / (b_idx + 1)]
        x_adv_val = batch_attack(imgs_val, labels_val, attack, fmodel, eps, batch_size)
        val_eval = result_model.evaluate(x_adv_val, y_val, verbose=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': val_eval[0],
                 'val_accuracy': val_eval[1]}
        print(i + nat_epochs, time.time() - delta0, 's', stats)
        for cb in callbacks:
            cb.on_epoch_end(i + nat_epochs, stats)
            if i == (adv_epochs - 1) or result_model.stop_training:
                cb.on_train_end()
        if result_model.stop_training:
            break
    return result_model


def main():
    #train_adv(MVGG4Builder(), MNIST(), epochs=bf.arg_flags.epoch, batch_size=bf.arg_flags.batch_size, pruning_fn=soft_prune_model2)
    train_nat(MVGG4Builder(), MNIST(), epochs=bf.arg_flags.epoch, batch_size=bf.arg_flags.batch_size, pruning_fn=soft_prune_model2)


if __name__ == '__main__':
    bf.arg_parser.add_argument("--batch_size", type=int, default=50)
    bf.arg_parser.add_argument("--epoch", type=int, default=80)
    bf.arg_parser.add_argument("--save_dir", type=str, default=os.path.join('saved_models'))
    bf.arg_parser.add_argument('--step_size', type=float, default=0.01)
    bf.arg_parser.add_argument('--steps', type=int, default=40)
    bf.arg_parser.add_argument('--eps', type=float, default=0.3)

    bf.setup_base_frame(main)
