import torch
import torchvision
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from intervaltree import IntervalTree
from models import load_model
from activations import get_activations, get_inactivity_ratio, get_activity


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


def load_data(path):
    return torch.load(path)


def split_dataset(dataset, split=0.1, seed=42):
    second_partition_size = int(len(dataset) * split)
    return torch.utils.data.random_split(
        dataset,
        [len(dataset) - second_partition_size, second_partition_size],
        torch.Generator().manual_seed(seed)
    )


def evaluate(model, loss_fn, ds_loader, attack=None):
    model_device = next(model.parameters()).device
    loss_sum = 0
    acc_sum = 0
    for i, (x, y) in enumerate(ds_loader):
        x, y = x.to(model_device), y.to(model_device)

        if attack is not None:
            x = attack(x, y)
        with torch.no_grad():
            output = model(x)

            loss = loss_fn(output, y)

        loss_sum += loss.item()
        acc_sum += np.mean(torch.argmax(output, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy())
    return loss_sum / (i + 1), acc_sum / (i + 1)


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = IntervalTree()
        cumulative_length = 0
        for d in datasets:
            self.datasets.addi(cumulative_length, cumulative_length + len(d), d)
            cumulative_length += len(d)

    def __getitem__(self, i):
        # Getting the one element from the set since it will always contain only one
        # (because there can not be overlapping in the intervals)
        dataset_interval = next(iter(self.datasets[i]))
        return dataset_interval.data[i - dataset_interval.begin]

    def __len__(self):
        cumulative_length = 0
        for d in self.datasets:
            cumulative_length += len(d.data)
        return cumulative_length


class LogManager:
    def __init__(self, to_file=None, from_file=None):
        self.file = to_file
        self.records = []
        if from_file is not None:
            with open(from_file, 'r') as fp:
                for record in fp.readlines():
                    self.records.append(json.loads(record))
            if self.file is None:
                self.file = from_file

    def write_record(self, record):
        self.records.append(record)
        if self.file is not None:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file, 'a') as fp:
                fp.write(json.dumps(self.records[-1]) + '\n')

    def write_all(self):
        assert self.file is not None, 'A file must be provided in order to write!'
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'w') as fp:
            for r in self.records:
                fp.write(json.dumps(r) + '\n')

    def get_transposed(self):
        assert len(self.records) > 0, 'There must be at least one record in order to transpose!'

        transposed_metrics = {}
        for key in self.records[0].keys():
            transposed_metrics[key] = []

        for rec in self.records:
            assert rec.keys() == transposed_metrics.keys(), 'All record keys must be the same in order to transpose!'

        for rec in self.records:
            for k, v in rec.items():
                transposed_metrics[k].append(v)
        return transposed_metrics

    def has_col(self, col_name):
        ret = True
        for rec in self.records:
            ret &= col_name in rec
        return ret


def append_activations_to_log(log_file, checkpoint_dir, model, ds_loader, attack=None, epoch_interval=None, sampling=1, force=False):
    # epoch_interval tuple inclusion: [first, last)
    lm = LogManager(from_file=log_file)

    check_std_activations = force or not lm.has_col('std_inactivity_ratio')
    check_adv_activations = (force or not lm.has_col('adv_inactivity_ratio')) and attack is not None
    if not check_std_activations and not check_adv_activations:
        return

    if epoch_interval is None:
        epoch_interval = (1, len(lm.records) + 1)

    std_inactivity_ratio = np.array([])
    adv_inactivity_ratio = np.array([])
    for i in range(*epoch_interval):
        if i % sampling == 0:
            load_model(model, f'{checkpoint_dir}/{i}')
            if check_std_activations:
                #lm.records[i - 1]['std_inactivity_ratio'] = get_inactivity_ratio(get_activations(model, model.get_relevant_layers(), ds_loader))
                std_inactivity_ratio = np.append(std_inactivity_ratio, get_inactivity_ratio(get_activations(model, model.get_relevant_layers(), ds_loader)))
            if check_adv_activations:
                #lm.records[i - 1]['adv_inactivity_ratio'] = get_inactivity_ratio(get_activations(model, model.get_relevant_layers(), ds_loader, attack))
                adv_inactivity_ratio = np.append(adv_inactivity_ratio, get_inactivity_ratio(get_activations(model, model.get_relevant_layers(), ds_loader, attack)))
        else:
            if check_std_activations:
                std_inactivity_ratio = np.append(std_inactivity_ratio, np.nan)
            if check_adv_activations:
                adv_inactivity_ratio = np.append(adv_inactivity_ratio, np.nan)

    if check_std_activations:
        xp = (~np.isnan(std_inactivity_ratio)).ravel().nonzero()[0]
        fp = std_inactivity_ratio[~np.isnan(std_inactivity_ratio)]
        x = np.isnan(std_inactivity_ratio).ravel().nonzero()[0]
        std_inactivity_ratio[np.isnan(std_inactivity_ratio)] = np.interp(x, xp, fp)
    if check_adv_activations:
        xp = (~np.isnan(adv_inactivity_ratio)).ravel().nonzero()[0]
        fp = adv_inactivity_ratio[~np.isnan(adv_inactivity_ratio)]
        x = np.isnan(adv_inactivity_ratio).ravel().nonzero()[0]
        adv_inactivity_ratio[np.isnan(adv_inactivity_ratio)] = np.interp(x, xp, fp)

    for i in range(*epoch_interval):
        if check_std_activations:
            lm.records[i - 1]['std_inactivity_ratio'] = std_inactivity_ratio[i - epoch_interval[0]]
        if check_adv_activations:
            lm.records[i - 1]['adv_inactivity_ratio'] = adv_inactivity_ratio[i - epoch_interval[0]]

    lm.write_all()


def simplify_tdict(td, device=None):  # td = tensor dict
    if device is None:
        device = next(iter(td.values())).device
    ret = torch.tensor([], device=device)
    for v in td.values():
        ret = torch.cat((ret, torch.flatten(v)), dim=0)
    return ret


def generate_activities(model, path, epoch_range, training_methods, attachments, ds_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    architecture = str.split('/')[-1]
    for training_method in training_methods:
        for attachment in attachments:
            for i in epoch_range:
                print(f'{path}/{training_method}_{architecture}{attachment}/{i}')
                load_model(model, f'{path}/{training_method}_{architecture}{attachment}/{i}')
                std_activity = get_activity(model, model.get_relevant_layers(), ds_loader)
                save_data(std_activity, f'{path}/activities/{training_method}_{architecture}{attachment}/{i}')


class Regularization:
    def __init__(self, model=None, lam=0.0):
        self.model = model
        self.lam = lam

    def norm(self):
        return 0.0

    def __call__(self):
        return self.norm() * self.lam


class L2Regularization(Regularization):
    def __init__(self, model, l2_lambda):
        super(L2Regularization, self).__init__(model, l2_lambda)

    def norm(self):
        return sum(p.pow(2.0).sum() for p in self.model.parameters())


class L1Regularization(Regularization):
    def __init__(self, model, l1_lambda):
        super(L1Regularization, self).__init__(model, l1_lambda)

    def norm(self):
        return sum(p.abs().sum() for p in self.model.parameters())


class LbRegularization(Regularization):
    def __init__(self, model_parameters, lb_lambda):
        super(LbRegularization, self).__init__(model_parameters, lb_lambda)

    def norm(self):
        return sum(((p + p.abs()) / 2).pow(2.0).sum() for p in self.model.parameters())


def load_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    combined = []
    train_dataset = torchvision.datasets.MNIST('./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=6)
    combined.append(train_dataset)
    test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=6)
    combined.append(test_dataset)

    # train_dataset, val_dataset = split_dataset(train_dataset, val_split)
    # val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=6)
    # combined.append(val_dataset)

    combined_dataset = MultiDataset(*combined)
    combined_loader = DataLoader(combined_dataset, batch_size=50, shuffle=True, num_workers=6)

    return train_loader, test_loader, combined_loader


def load_cifar10():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    combined = []
    train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
    combined.append(train_dataset)
    test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6)
    combined.append(test_dataset)

    combined_dataset = MultiDataset(*combined)
    combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=6)

    return train_loader, test_loader, combined_loader


def load_fashion_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    combined = []
    train_dataset = torchvision.datasets.FashionMNIST('./datasets', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=6)
    combined.append(train_dataset)
    test_dataset = torchvision.datasets.FashionMNIST('./datasets', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=6)
    combined.append(test_dataset)

    combined_dataset = MultiDataset(*combined)
    combined_loader = DataLoader(combined_dataset, batch_size=50, shuffle=True, num_workers=6)

    return train_loader, test_loader, combined_loader
