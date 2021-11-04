import torch
import torch.nn as nn


class VGG4(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG4, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)  # This is important for the neuron activation checks!

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.conv2d_1 = nn.Conv2d(32, 64, (5, 5), padding='same')
        self.linear_0 = nn.Linear(64 * (input_shape[1] // (2 * 2) * input_shape[2] // (2 * 2)), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv2d_0(x)))
        x = self.max_pool(self.relu(self.conv2d_1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x


class WideVGG4(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(WideVGG4, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 64, (5, 5), padding='same')
        self.conv2d_1 = nn.Conv2d(64, 128, (5, 5), padding='same')
        self.linear_0 = nn.Linear(128 * (input_shape[1] // (2 * 2) * input_shape[2] // (2 * 2)), 2048)
        self.linear_1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv2d_0(x)))
        x = self.max_pool(self.relu(self.conv2d_1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x