import torch
import numpy as np
from time import time
from collections import deque
from utils import evaluate
from models import save_checkpoint


def train_adv(model, loss_fn, opt, attack, train_loader, val_loader=None,
              checkpoint_dir=None, metrics_lm=None, num_epochs=1):
    device = next(model.parameters()).device
    model.training = True
    log_metrics = metrics_lm is not None
    num_batches = len(train_loader)

    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        metrics = {}
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            x = attack(x, y)
            output = model(x)

            loss = loss_fn(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(np.mean(torch.argmax(output, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy()))
            reduced_batch_losses = np.mean(batch_losses)
            reduced_batch_accs = np.mean(batch_accs)
            print(f'\r  {i + 1}/{num_batches} {time() - begin_time:.2f}s - adv_train_loss: {reduced_batch_losses:.4f}, adv_train_acc: {reduced_batch_accs:.4f}', end='')
        metrics['adv_train_loss'] = reduced_batch_losses
        metrics['adv_train_acc'] = reduced_batch_accs
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader, attack)
            print(f', adv_val_loss: {val_loss:.4f}, adv_val_acc: {val_acc:.4f}', end='')
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            print(f', std_val_loss: {val_loss:.4f}, std_val_acc: {val_acc:.4f}', end='')
            metrics['adv_val_loss'] = val_loss
            metrics['adv_val_acc'] = val_acc
        if log_metrics:
            metrics_lm.write_record(metrics)
        if checkpoint_dir is not None:
            save_checkpoint(model, opt, f'{checkpoint_dir}/{epoch_idx + 1}')
        print()
    model.training = False


def train_dynamic_hybrid(model, loss_fn, opt, attack, train_loader, val_loader=None,
                         checkpoint_dir=None, metrics_lm=None, loss_window=5, loss_deviation=0.05, num_epochs=1):
    device = next(model.parameters()).device
    model.training = True
    log_metrics = metrics_lm is not None
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

            loss = loss_fn(output, y)

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
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader, attack)
            print(f', adv_val_loss: {val_loss:.4f}, adv_val_acc: {val_acc:.4f}', end='')
            metrics['adv_val_loss'] = val_loss
            metrics['adv_val_acc'] = val_acc
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            print(f', std_val_loss: {val_loss:.4f}, std_val_acc: {val_acc:.4f}', end='')
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
        if log_metrics:
            metrics_lm.write_record(metrics)
        if checkpoint_dir is not None:
            save_checkpoint(model, opt, f'{checkpoint_dir}/{epoch_idx + 1}')
        if not switched and epoch_idx >= loss_window and np.std([np.mean(epoch_losses), reduced_batch_losses]) < loss_deviation:
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        epoch_losses.appendleft(reduced_batch_losses)
        print()
    model.training = False


def train_static_hybrid(model, loss_fn, opt, attack, train_loader, val_loader=None,
                        checkpoint_dir=None, metrics_lm=None, switch_point=1, num_epochs=1):
    device = next(model.parameters()).device
    model.training = True
    log_metrics = metrics_lm is not None
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

            loss = loss_fn(output, y)

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
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader, attack)
            print(f', adv_val_loss: {val_loss:.4f}, adv_val_acc: {val_acc:.4f}', end='')
            metrics['adv_val_loss'] = val_loss
            metrics['adv_val_acc'] = val_acc
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            print(f', std_val_loss: {val_loss:.4f}, std_val_acc: {val_acc:.4f}', end='')
            metrics['std_val_loss'] = val_loss
            metrics['std_val_acc'] = val_acc
        if log_metrics:
            metrics_lm.write_record(metrics)
        if checkpoint_dir is not None:
            save_checkpoint(model, opt, f'{checkpoint_dir}/{epoch_idx + 1}')
        if not switched and epoch_idx >= (switch_point - 1):
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        print()
    model.training = False
