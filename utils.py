import torch
import numpy as np
import json
import os
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


class MetricsLogManager:
    def __init__(self, to_file=None, from_file=None):
        self.file = to_file
        self.records = []
        if from_file is not None:
            with open(from_file, 'r') as fp:
                for record in fp.readlines():
                    self.records.append(json.loads(record))

    def write_record(self, record):
        self.records.append(record)
        if self.file is not None:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file, 'a') as fp:
                fp.write(json.dumps(self.records[-1]) + '\n')

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


class Regularization:
    def __init__(self, model_parameters=None, lam=0.0):
        self.model_parameters = model_parameters
        self.lam = lam

    def norm(self):
        return 0.0

    def __call__(self):
        return self.norm() * self.lam


class L2Regularization(Regularization):
    def __init__(self, model_parameters, l2_lambda):
        super(L2Regularization, self).__init__(model_parameters, l2_lambda)

    def norm(self):
        return sum(p.pow(2.0).sum() for p in self.model_parameters)


class L1Regularization(Regularization):
    def __init__(self, model_parameters, l1_lambda):
        super(L1Regularization, self).__init__(model_parameters, l1_lambda)

    def norm(self):
        return sum(p.abs().sum() for p in self.model_parameters)


class LbRegularization(Regularization):
    def __init__(self, model_parameters, lb_lambda):
        super(LbRegularization, self).__init__(model_parameters, lb_lambda)

    def norm(self):
        return sum(((p + p.abs()) / 2).pow(2.0).sum() for p in self.model_parameters)
