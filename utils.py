import torch
from intervaltree import IntervalTree


def evaluate(pred_y, y):
    return (pred_y == y).mean()


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = IntervalTree()
        prev_d_len = 0
        for d in datasets:
            self.datasets.addi(prev_d_len, prev_d_len + len(d), d)
            prev_d_len = len(d)

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
