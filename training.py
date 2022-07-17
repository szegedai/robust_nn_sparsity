import torch
import numpy as np
import sys
import os
import paramiko
import wandb
from time import time
from collections import deque


from utils import evaluate, WeightRegularization, RollingStatistics, LogManager
from models import save_checkpoint, freeze_layers, unfreeze_layers


class Callback:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def on_training_begin(self, training_vars):
        pass

    def on_training_end(self, training_vars):
        pass

    def on_epoch_begin(self, training_vars):
        pass

    def on_epoch_end(self, training_vars):
        pass

    def on_batch_begin(self, training_vars):
        pass

    def on_batch_end(self, training_vars):
        pass


class CLILoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_begin_time = None
        self.training_begin_time = None

    def on_training_begin(self, training_vars):
        self.training_begin_time = time()

    def on_epoch_begin(self, training_vars):
        self.epoch_begin_time = time()
        print(f'{training_vars["epoch_idx"] + 1}/{training_vars["num_epochs"]} epoch:\n')

    def on_epoch_end(self, training_vars):
        sys.stdout.write('\x1b[1A\x1b[2K')
        print(f'  {time() - self.epoch_begin_time:.2f}s - {self.metrics_to_str(training_vars["metrics"])}')

    def on_batch_end(self, training_vars):
        sys.stdout.write('\x1b[1A\x1b[2K')
        print(f'  {training_vars["batch_idx"] + 1}/{training_vars["num_batches"]} {time() - self.epoch_begin_time:.2f}s -', self.metrics_to_str(training_vars["metrics"]))

    @staticmethod
    def metrics_to_str(m: dict):
        ret = ''
        for k, v in m.items():
            ret += f'{k}: {v:.4f} '
        return ret


class LRSchedulerCallback(Callback):
    def __init__(self, schedular):
        super().__init__()
        self.schedular = schedular

    def on_epoch_end(self, training_vars):
        self.schedular.step()


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, ever_n_epochs=1):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.ever_n_epochs = ever_n_epochs

    def on_epoch_end(self, training_vars):
        epoch_idx = training_vars['epoch_idx']
        if not ((epoch_idx + 1) % self.ever_n_epochs):
            save_checkpoint(self.model, self.optimizer, f'{self.checkpoint_dir}/{epoch_idx + 1}')


class RemoteCheckpointCallback(Callback):
    def __init__(self, host, username, password, remote_dir, tmp_dir='.', ever_n_epochs=1):
        super().__init__()
        self.host = host
        self.username = username
        self.password = password
        self.remote_dir = remote_dir
        self.tmp_dir = tmp_dir
        self.ever_n_epochs = ever_n_epochs
        self._transport = paramiko.Transport((host, 22))

    def on_training_begin(self, training_vars):
        self._transport.connect(username=self.username, password=self.password)

    def on_training_end(self, training_vars):
        self._transport.close()

    def on_epoch_end(self, training_vars):
        epoch_idx = training_vars['epoch_idx']
        if not ((epoch_idx + 1) % self.ever_n_epochs):
            save_checkpoint(self.model, self.optimizer, f'{self.tmp_dir}/tmp_{epoch_idx + 1}')
            with paramiko.SFTPClient.from_transport(self._transport) as sftp:
                sftp.put(os.path.abspath(f'{self.tmp_dir}/tmp_{epoch_idx + 1}'), f'{self.remote_dir}/{epoch_idx + 1}')
            os.remove(f'{self.tmp_dir}/tmp_{epoch_idx + 1}')


class WandBLoggerCallback(Callback):
    def __init__(self, project_name, name, config, **kwargs):
        super().__init__()
        wandb.init(project=project_name, config=config, name=name, **kwargs)

    def on_epoch_end(self, training_vars):
        wandb.log({'epoch': training_vars['epoch_idx'] + 1, **training_vars['metrics']})

    def on_training_end(self, training_vars):
        wandb.finish(0)


class FileLoggerCallback(Callback):
    def __init__(self, to_file, from_file=None):
        super().__init__()
        self.log_manager = LogManager(to_file, from_file)

    def on_epoch_end(self, training_vars):
        self.log_manager.write_record(training_vars['metrics'])


class LayerFreezerCallback(Callback):
    def __init__(self, layers, use_complementer_layers=False):
        super().__init__()
        self.layers = layers
        self.use_complementer_layers = use_complementer_layers

    def on_epoch_begin(self, training_vars):
        if self.use_complementer_layers:
            freeze_layers(self.model)
            unfreeze_layers(self.model, self.layers)
        else:
            freeze_layers(self.model, self.layers)


class CallbackInserterCallback(Callback):
    def __init__(self, epoch_idx, callbacks):
        super().__init__()
        self.epoch_idx = epoch_idx
        self.callbacks = callbacks

    def on_epoch_begin(self, training_vars):
        if self.epoch_idx == training_vars['epoch_idx']:
            training_vars['callbacks'] += self.callbacks


class AdversarialPerturbationCallback(Callback):
    def __init__(self, attack):
        super().__init__()
        self.attack = attack

    def on_batch_begin(self, training_vars):
        #x.copy_(self.attack.perturb(x, y).detach_())
        x = training_vars['x']
        y = training_vars['y']
        self.model.eval()
        x.copy_(self.attack.perturb(x, y))
        self.model.train()


def train_classifier(model, loss_fn, opt, train_loader, val_loader=None, val_attack=None,
                     regs=None, callbacks=None, num_epochs=1, initial_epoch=1):
    assert num_epochs >= 1
    assert initial_epoch >= 1
    initial_epoch -= 1
    device = next(model.parameters()).device
    callbacks = callbacks if callbacks is not None else []
    regs = regs if regs is not None else []
    do_validation = val_loader is not None
    do_adv_validation = val_attack is not None
    num_batches = len(train_loader)
    running_loss = RollingStatistics()
    running_acc = RollingStatistics()

    for callback in callbacks:
        callback.model = model
        callback.optimizer = opt

    def iter_callbacks(callback_name, training_vars):
        for c in callbacks:
            getattr(c, callback_name)(training_vars)

    iter_callbacks('on_training_begin', locals())
    for epoch_idx in range(initial_epoch, initial_epoch + num_epochs):
        model.train()
        metrics = {}
        running_loss.reset()
        running_acc.reset()
        iter_callbacks('on_epoch_begin', locals())
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            iter_callbacks('on_batch_begin', locals())

            output = model(x)

            loss = loss_fn(output, y) + sum(reg() for reg in regs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss.update([loss.item()])
            running_acc.update(torch.argmax(output, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy())
            reduced_batch_losses = running_loss.mean
            reduced_batch_accs = running_acc.mean
            metrics['train_loss'] = reduced_batch_losses
            metrics['train_acc'] = reduced_batch_accs
            iter_callbacks('on_batch_end', locals())
        if do_validation:
            model.eval()
            if do_adv_validation:
                val_loss, val_acc = evaluate(model, loss_fn, val_loader, val_attack)
                metrics['adv_val_loss'] = val_loss
                metrics['adv_val_acc'] = val_acc
                #iter_callbacks('on_adv_validation', metrics)
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
            #iter_callbacks('on_std_validation', metrics)
        iter_callbacks('on_epoch_end', locals())
    iter_callbacks('on_training_end', locals())
    model.eval()
    return metrics


def train_regression(model, loss_fn, opt, train_loader, callbacks=None, num_epochs=1):
    device = next(model.parameters()).device
    callbacks = callbacks if callbacks is not None else []
    num_batches = len(train_loader)
    running_loss = RollingStatistics()

    for callback in callbacks:
        callback.model = model
        callback.optimizer = opt

    def iter_callbacks(callback_name, *args, **kwargs):
        for c in callbacks:
            getattr(c, callback_name)(*args, **kwargs)

    iter_callbacks('on_training_begin')
    for epoch_idx in range(num_epochs):
        model.train()
        metrics = {}
        running_loss.reset()
        iter_callbacks('on_epoch_begin', num_epochs, epoch_idx)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            iter_callbacks('on_batch_begin', num_batches, batch_idx, x, y)

            output = model(x)

            loss = loss_fn(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss.update([loss.item()])
            reduced_batch_losses = running_loss.mean
            metrics['std_train_loss'] = reduced_batch_losses
            iter_callbacks('on_batch_end', num_batches, batch_idx, metrics)
        iter_callbacks('on_epoch_end', num_epochs, epoch_idx, metrics)
    iter_callbacks('on_training_end', metrics)
    model.eval()
    return metrics



'''def train_dynamic_hybrid(model, loss_fn, opt, attack, train_loader, val_loader=None, val_attack=None,
                         reg=WeightRegularization(), checkpoint_dir=None, log_manager=None,
                         loss_window=5, loss_deviation=0.05, num_epochs=1):
    device = next(model.parameters()).device
    model.training = True
    log_metrics = log_manager is not None
    do_validation = val_loader is not None
    make_checkpoints = checkpoint_dir is not None
    if val_attack is None:
        val_attack = attack
    num_batches = len(train_loader)

    switched = False
    # For example, in case of Adam:
    # Does not include lr, bets, eps and weight_decay.
    # 'state' only stores the momentums.
    opt_init_state = opt.state

    epoch_losses = deque([], loss_window)
    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        metrics = {}
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if switched:
                x = attack(x, y)
            output = model(x)

            loss = loss_fn(output, y) + reg()

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(np.mean(torch.argmax(output, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy()))
            reduced_batch_losses = np.mean(batch_losses)
            reduced_batch_accs = np.mean(batch_accs)
            if switched:
                print(f'\r  {i + 1}/{num_batches} {time() - begin_time:.2f}s - adv_train_loss: {reduced_batch_losses:.4f}, adv_train_acc: {reduced_batch_accs:.4f}', end='')
            else:
                print(f'\r  {i + 1}/{num_batches} {time() - begin_time:.2f}s - std_train_loss: {reduced_batch_losses:.4f}, std_train_acc: {reduced_batch_accs:.4f}', end='')
        if switched:
            metrics['adv_train_loss'] = reduced_batch_losses
            metrics['adv_train_acc'] = reduced_batch_accs
        else:
            metrics['std_train_loss'] = reduced_batch_losses
            metrics['std_train_acc'] = reduced_batch_accs
        if do_validation:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader, val_attack)
            print(f', adv_val_loss: {val_loss:.4f}, adv_val_acc: {val_acc:.4f}', end='')
            metrics['adv_val_loss'] = val_loss
            metrics['adv_val_acc'] = val_acc
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            print(f', std_val_loss: {val_loss:.4f}, std_val_acc: {val_acc:.4f}', end='')
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
        if log_metrics:
            log_manager.write_record(metrics)
        if make_checkpoints:
            save_checkpoint(model, opt, f'{checkpoint_dir}/{epoch_idx + 1}')
        if not switched and epoch_idx >= loss_window and np.std([np.mean(epoch_losses), reduced_batch_losses]) < loss_deviation:
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        epoch_losses.appendleft(reduced_batch_losses)
        print()
    model.training = False


def train_static_hybrid(model, loss_fn, opt, attack, train_loader, val_loader=None, val_attack=None,
                        reg=WeightRegularization(), checkpoint_dir=None, log_manager=None,
                        switch_point=1, num_epochs=1):
    device = next(model.parameters()).device
    model.training = True
    log_metrics = log_manager is not None
    do_validation = val_loader is not None
    make_checkpoints = checkpoint_dir is not None
    if val_attack is None:
        val_attack = attack
    num_batches = len(train_loader)

    switched = False
    # For example, in case of Adam:
    # Does not include lr, bets, eps and weight_decay.
    # 'state' only stores the momentums.
    opt_init_state = opt.state

    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        metrics = {}
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if switched:
                x = attack(x, y)
            output = model(x)

            loss = loss_fn(output, y) + reg()

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(np.mean(torch.argmax(output, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy()))
            reduced_batch_losses = np.mean(batch_losses)
            reduced_batch_accs = np.mean(batch_accs)
            if switched:
                print(f'\r  {i + 1}/{num_batches} {time() - begin_time:.2f}s - adv_train_loss: {reduced_batch_losses:.4f}, adv_train_acc: {reduced_batch_accs:.4f}', end='')
            else:
                print(f'\r  {i + 1}/{num_batches} {time() - begin_time:.2f}s - std_train_loss: {reduced_batch_losses:.4f}, std_train_acc: {reduced_batch_accs:.4f}', end='')
        if switched:
            metrics['adv_train_loss'] = reduced_batch_losses
            metrics['adv_train_acc'] = reduced_batch_accs
        else:
            metrics['std_train_loss'] = reduced_batch_losses
            metrics['std_train_acc'] = reduced_batch_accs
        if do_validation:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader, val_attack)
            print(f', adv_val_loss: {val_loss:.4f}, adv_val_acc: {val_acc:.4f}', end='')
            metrics['adv_val_loss'] = val_loss
            metrics['adv_val_acc'] = val_acc
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            print(f', std_val_loss: {val_loss:.4f}, std_val_acc: {val_acc:.4f}', end='')
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
        if log_metrics:
            log_manager.write_record(metrics)
        if make_checkpoints:
            save_checkpoint(model, opt, f'{checkpoint_dir}/{epoch_idx + 1}')
        if not switched and epoch_idx >= (switch_point - 1):
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        print()
    model.training = False'''
