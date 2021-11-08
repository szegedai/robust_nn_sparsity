import torch
import numpy as np
from intervaltree import IntervalTree


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
