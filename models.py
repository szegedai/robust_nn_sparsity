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


class VGG2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG2, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.linear_0 = nn.Linear(32 * (input_shape[1] // 2 * input_shape[2] // 2), num_classes)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv2d_0(x)))
        x = torch.flatten(x, 1)
        x = self.linear_0(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['conv2d_0']


class VGG2BN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG2BN, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.bn_0 = nn.BatchNorm2d(32)
        self.linear_0 = nn.Linear(32 * (input_shape[1] // 2 * input_shape[2] // 2), num_classes)

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn_0(self.conv2d_0(x))))
        x = torch.flatten(x, 1)
        x = self.linear_0(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['bn_0']


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


class VGG4BN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG4BN, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)  # This is important for the neuron activation checks!

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.bn_0 = nn.BatchNorm2d(32)
        self.conv2d_1 = nn.Conv2d(32, 64, (5, 5), padding='same')
        self.bn_1 = nn.BatchNorm2d(64)
        self.linear_0 = nn.Linear(64 * (input_shape[1] // (2 * 2) * input_shape[2] // (2 * 2)), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.bn_0(x)
        x = self.max_pool(self.relu(x))
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.max_pool(self.relu(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['bn_0', 'bn_1', 'linear_0']


class VGG6(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG6, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.conv2d_1 = nn.Conv2d(32, 32, (5, 5), padding='same')
        self.conv2d_2 = nn.Conv2d(32, 64, (5, 5), padding='same')
        self.conv2d_3 = nn.Conv2d(64, 64, (5, 5), padding='same')
        self.linear_0 = nn.Linear(64 * (input_shape[1] // (2 * 2) * input_shape[2] // (2 * 2)), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.conv2d_0(x))
        x = self.relu(self.conv2d_1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['conv2d_0', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'linear_0']


class VGG6BN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG6BN, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.bn_0 = nn.BatchNorm2d(32)
        self.conv2d_1 = nn.Conv2d(32, 32, (5, 5), padding='same')
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(32, 64, (5, 5), padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, (5, 5), padding='same')
        self.bn_3 = nn.BatchNorm2d(64)
        self.linear_0 = nn.Linear(64 * (input_shape[1] // (2 * 2) * input_shape[2] // (2 * 2)), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.relu(self.bn_0(x))
        x = self.conv2d_1(x)
        x = self.relu(self.bn_1(x))
        x = self.max_pool(x)
        x = self.conv2d_2(x)
        x = self.relu(self.bn_2(x))
        x = self.conv2d_3(x)
        x = self.relu(self.bn_3(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['bn_0', 'bn_1', 'bn_2', 'bn_3', 'linear_0']


class VGG19(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG19, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['E'], batch_norm=False),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG19, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.0', 'features.2', 'features.5', 'features.7',
                'features.10', 'features.12', 'features.14', 'features.16',
                'features.19', 'features.21', 'features.23', 'features.25',
                'features.28', 'features.30', 'features.32', 'features.34',
                'classifier.0', 'classifier.3', 'classifier.6']


class VGG19BN(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG19BN, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['E'], batch_norm=True),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG19BN, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.1', 'features.4', 'features.8', 'features.11',
                'features.15', 'features.18', 'features.21', 'features.24',
                'features.28', 'features.31', 'features.34', 'features.37',
                'features.41', 'features.44', 'features.47', 'features.50',
                'classifier.0', 'classifier.3', 'classifier.6']


class ResNet18(torchvision.models.ResNet):
    def __init__(self, num_classes, **kwargs):
        super(ResNet18, self).__init__(
            torchvision.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            num_classes,
            **kwargs
        )

    def forward(self, x):
        return super(ResNet18, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['body.bn1',
                'body.layer1.0.bn1', 'body.layer1.0.bn2', 'body.layer1.1.bn1', 'body.layer1.1.bn2',
                'body.layer2.0.bn1', 'body.layer2.0.bn2', 'body.layer2.1.bn1', 'body.layer2.1.bn2',
                'body.layer3.0.bn1', 'body.layer3.0.bn2', 'body.layer3.1.bn1', 'body.layer3.1.bn2',
                'body.layer4.0.bn1', 'body.layer4.0.bn2', 'body.layer4.1.bn1', 'body.layer4.1.bn2',
                'body.fc']
