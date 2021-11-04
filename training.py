import torch
import numpy as np
from time import time
from collections import deque
from utils import evaluate


def train_adv(model, loss_fn, opt, train_loader,  attack, num_epochs=1, device=None):
    device = device if device else torch.device('cpu')
    model.training = True

    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            x = attack(x, y, True)
            output = model(x)

            loss = loss_fn(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(evaluate(torch.argmax(output, dim=1).detach().cpu().numpy(), y.detach().cpu().numpy()))
            print(f'\r  {i + 1} batch {time() - begin_time:.2f}s - loss: {np.mean(batch_losses):.4f}, adv_acc: {np.mean(batch_accs):.4f}', end='')
        print()
    model.training = False


def train_dynamic_hybrid(model, loss_fn, opt, train_loader, attack, loss_window, loss_deviation, num_epochs=1, device=None):
    device = device if device else torch.device('cpu')
    model.training = True

    switched = False
    # For example, in case of Adam:
    # Does not include lr, bets, eps and weight_decay.
    # 'state' only stores the momentums.
    opt_init_state = opt.state

    epoch_losses = deque([], loss_window)
    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if switched:
                x = attack(x, y, True)
            output = model(x)

            loss = loss_fn(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(evaluate(torch.argmax(output, dim=1).detach().cpu().numpy(), y.detach().cpu().numpy()))
            reduced_batch_losses = np.mean(batch_losses)
            reduced_batch_accs = np.mean(batch_accs)
            print(f'\r  {i + 1} batch {time() - begin_time:.2f}s - loss: {reduced_batch_losses:.4f}, acc: {reduced_batch_accs:.4f}', end='')
        if not switched and epoch_idx >= loss_window and np.std([np.mean(epoch_losses), reduced_batch_losses]) < loss_deviation:
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        epoch_losses.appendleft(reduced_batch_losses)
        print()
    model.training = False


def train_static_hybrid(model, loss_fn, opt, train_loader, attack, switch_point, num_epochs=1, device=None):
    device = device if device else torch.device('cpu')
    model.training = True

    switched = False
    # For example, in case of Adam:
    # Does not include lr, bets, eps and weight_decay.
    # 'state' only stores the momentums.
    opt_init_state = opt.state

    for epoch_idx in range(num_epochs):
        print(f'{epoch_idx + 1}/{num_epochs} epoch:')
        begin_time = time()
        batch_losses = []
        batch_accs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if switched:
                x = attack(x, y, True)
            output = model(x)

            loss = loss_fn(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())
            batch_accs.append(evaluate(torch.argmax(output, dim=1).detach().cpu().numpy(), y.detach().cpu().numpy()))
            reduced_batch_losses = np.mean(batch_losses)
            reduced_batch_accs = np.mean(batch_accs)
            print(f'\r  {i + 1} batch {time() - begin_time:.2f}s - loss: {reduced_batch_losses:.4f}, acc: {reduced_batch_accs:.4f}', end='')
        if not switched and epoch_idx >= (switch_point - 1):
            switched = True
            # Reinitialize optimizer inner state on switching.
            # In case of Adam, the momentums are reinitialized.
            opt.state = opt_init_state
        print()
    model.training = False