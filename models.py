import torch
import torch.nn as nn
import torchvision
import os


def save_checkpoint(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict()}, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path)['model_state'])


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

    @staticmethod
    def get_relevant_layers():
        return ['conv2d_0', 'conv2d_1', 'linear_0']


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

    @staticmethod
    def get_relevant_layers():
        return ['conv2d_0', 'conv2d_1', 'linear_0']


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.body = torchvision.models.vgg19()
        self.out_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.out_layer(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['body.features.0', 'body.features.2', 'body.features.5', 'body.features.7',
                'body.features.10', 'body.features.12', 'body.features.14', 'body.features.16',
                'body.features.19', 'body.features.21', 'body.features.23', 'body.features.25',
                'body.features.28', 'body.features.30', 'body.features.32', 'body.features.34',
                'body.classifier.0', 'body.classifier.3', 'body.classifier.6']


class VGG19BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG19BN, self).__init__()
        self.body = torchvision.models.vgg19_bn()
        self.out_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.out_layer(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['body.features.1', 'body.features.4', 'body.features.8', 'body.features.11',
                'body.features.15', 'body.features.18', 'body.features.21', 'body.features.24',
                'body.features.28', 'body.features.31', 'body.features.34', 'body.features.37',
                'body.features.41', 'body.features.44', 'body.features.47', 'body.features.50',
                'body.classifier.0', 'body.classifier.3', 'body.classifier.6']


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.body = torchvision.models.resnet18()
        self.out_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = self.out_layer(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['body.conv1',
                'body.layer1.0.conv1', 'body.layer1.0.conv2', 'body.layer1.1.conv1', 'body.layer1.1.conv2',
                'body.layer2.0.conv1', 'body.layer2.0.conv2', 'body.layer2.1.conv1', 'body.layer2.1.conv2',
                'body.layer3.0.conv1', 'body.layer3.0.conv2', 'body.layer3.1.conv1', 'body.layer3.1.conv2',
                'body.layer4.0.conv1', 'body.layer4.0.conv2', 'body.layer4.1.conv1', 'body.layer4.1.conv2',
                'body.fc']
