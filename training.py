import torch
import numpy as np
import sys
import os
import paramiko
import wandb
from time import time
from collections import deque
from torch.utils.data import DataLoader


from utils import evaluate, WeightRegularization, RollingStatistics, LogManager, top1_accuracy
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


class MixupCallback(Callback):
    def __init__(self, alpha1=1., alpha2=None, dataloader=None):
        super().__init__()
        if alpha2 is None:
            alpha2 = alpha1
        self._beta_distribution = torch.distributions.beta.Beta(alpha1, alpha2)
        self._dl = dataloader
        self._dl_iter = None

    def on_training_begin(self, training_vars):
        if self._dl is None:
            train_loader = training_vars['train_loader']
            self._dl = DataLoader(train_loader.dataset,
                                  batch_size=train_loader.batch_size,
                                  shuffle=True,
                                  num_workers=train_loader.num_workers,
                                  pin_memory=train_loader.pin_memory)

    def on_epoch_begin(self, training_vars):
        self._dl_iter = iter(self._dl)

    def on_batch_begin(self, training_vars):
        x1 = training_vars['x']
        y1 = training_vars['y']
        lam = self._beta_distribution.sample((y1.shape[0],)).to(training_vars['device'])
        x2, y2 = next(self._dl_iter)
        x2, y2 = x2.to(x1.device), y2.to(y1.device)
        x1.copy_(lam.view(-1, 1, 1, 1) * x1 + (1. - lam.view(-1, 1, 1, 1)) * x2)
        y1.copy_(lam.view(-1, 1) * y1 + (1. - lam.view(-1, 1)) * y2)


class CutMixCallback(Callback):
    def __init__(self, alpha1=1., alpha2=None, dataloader=None):
        super().__init__()
        if alpha2 is None:
            alpha2 = alpha1
        self._beta_distribution = torch.distributions.beta.Beta(alpha1, alpha2)
        self._dl = dataloader
        self._dl_iter = None

    def on_training_begin(self, training_vars):
        if self._dl is None:
            train_loader = training_vars['train_loader']
            self._dl = DataLoader(train_loader.dataset,
                                  batch_size=train_loader.batch_size,
                                  shuffle=True,
                                  num_workers=train_loader.num_workers,
                                  pin_memory=train_loader.pin_memory)

    def on_epoch_begin(self, training_vars):
        self._dl_iter = iter(self._dl)

    def on_batch_begin(self, training_vars):
        x1 = training_vars['x']
        y1 = training_vars['y']
        lam = self._beta_distribution.sample((1,))
        x2, y2 = next(self._dl_iter)
        x2, y2 = x2.to(x1.device), y2.to(y1.device)
        w, h = x1.shape[3], x1.shape[2]
        c = torch.sqrt(1 - lam)
        cut_w, cut_h = (w * c).to(dtype=torch.int), (h * c).to(dtype=torch.int)
        center_x = torch.round(-w * torch.rand(1) + w).to(dtype=torch.int)
        center_y = torch.round(-h * torch.rand(1) + h).to(dtype=torch.int)
        cut_x1 = torch.clip(center_x - cut_w // 2, 0, w)
        cut_x2 = torch.clip(center_x + cut_w // 2, 0, w)
        cut_y1 = torch.clip(center_y - cut_h // 2, 0, h)
        cut_y2 = torch.clip(center_y + cut_h // 2, 0, h)
        lam = (1 - (cut_x2 - cut_x1) * (cut_y2 - cut_y1) / (w * h)).to(training_vars['device'])
        x1[:, :, cut_y1:cut_y2, cut_x1:cut_x2] = x2[:, :, cut_y1:cut_y2, cut_x1:cut_x2]
        y1.copy_(lam * y1 + (1. - lam) * y2)


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
                     regs=None, acc_fn=top1_accuracy(False), callbacks=None, num_epochs=1, initial_epoch=1):
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

            batch_acc = acc_fn(output, y)
            running_loss.update([loss.item()])
            running_acc.update(batch_acc)
            metrics['train_loss'] = running_loss.mean
            metrics['train_acc'] = running_acc.mean
            iter_callbacks('on_batch_end', locals())
        if do_validation:
            model.eval()
            if do_adv_validation:
                val_loss, val_acc = evaluate(model, val_loader, loss_fn, acc_fn, val_attack)
                metrics['adv_val_loss'] = val_loss
                metrics['adv_val_acc'] = val_acc
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, acc_fn)
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
        iter_callbacks('on_epoch_end', locals())
    iter_callbacks('on_training_end', locals())
    model.eval()
    return metrics
