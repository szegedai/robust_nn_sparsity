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


def load_checkpoint(model, optimizer, path, target_device=None):
    checkpoint = torch.load(path, target_device)
    model.load_state_dict(checkpoint['model_state'])
    if target_device is not None:
        model.to(target_device)
    optimizer.load_state_dict(checkpoint['optimizer_state'])


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict()}, path)


def load_model(model, path, target_device=None):
    model.load_state_dict(torch.load(path, target_device)['model_state'])
    if target_device is not None:
        model.to(target_device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: nn.Module, layers=None):
    for layer_name, layer in model.named_modules():
        if layers is None or layer_name in layers:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False


def unfreeze_layers(model: nn.Module, layers=None):
    for layer_name, layer in model.named_modules():
        if layers is None or layer_name in layers:
            for p in layer.parameters():
                p.requires_grad = True


class VGG4(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG4, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)  # Inplace is important for the neuron activation checks!

        self.conv2d_0 = nn.Conv2d(input_shape[0], 32, (5, 5), padding='same')
        self.conv2d_1 = nn.Conv2d(32, 64, (5, 5), padding='same')
        self.linear_0 = nn.Linear(64 * (input_shape[1] // 2 ** 2) * (input_shape[2] // 2 ** 2), 1024)
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
        self.linear_0 = nn.Linear(64 * (input_shape[1] // 2 ** 2) * (input_shape[2] // 2 ** 2), 1024)
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


class VGG10(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG10, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 64, (3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.conv2d_3 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.conv2d_6 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.linear_0 = nn.Linear(128 * (input_shape[1] // 2 ** 2) * (input_shape[2] // 2 ** 2), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.conv2d_0(x))
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2d_4(x))
        x = self.relu(self.conv2d_5(x))
        x = self.relu(self.conv2d_6(x))
        x = self.relu(self.conv2d_7(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['conv2d_0', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7',
                'linear_0']


class VGG10BN(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG10BN, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_0 = nn.Conv2d(input_shape[0], 64, (3, 3), padding='same')
        self.bn_0 = nn.BatchNorm2d(64)
        self.conv2d_1 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv2d_4 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.bn_4 = nn.BatchNorm2d(128)
        self.conv2d_5 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn_5 = nn.BatchNorm2d(128)
        self.conv2d_6 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn_6 = nn.BatchNorm2d(128)
        self.conv2d_7 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn_7 = nn.BatchNorm2d(128)
        self.linear_0 = nn.Linear(128 * (input_shape[1] // 2 ** 2) * (input_shape[2] // 2 ** 2), 1024)
        self.linear_1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.relu(self.bn_0(x))
        x = self.conv2d_1(x)
        x = self.relu(self.bn_1(x))
        x = self.conv2d_2(x)
        x = self.relu(self.bn_2(x))
        x = self.conv2d_3(x)
        x = self.relu(self.bn_3(x))
        x = self.max_pool(x)
        x = self.conv2d_4(x)
        x = self.relu(self.bn_4(x))
        x = self.conv2d_5(x)
        x = self.relu(self.bn_5(x))
        x = self.conv2d_6(x)
        x = self.relu(self.bn_6(x))
        x = self.conv2d_7(x)
        x = self.relu(self.bn_7(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.linear_0(x))
        x = self.linear_1(x)
        return x

    @staticmethod
    def get_relevant_layers():
        return ['bn_0', 'bn_1', 'bn_2', 'bn_3', 'bn_4', 'bn_5', 'bn_6', 'bn_7',
                'linear_0']


'''class VGG11(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG11, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['A'], batch_norm=False),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG11, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.0', 'features.3', 'features.6', 'features.8',
                'features.11', 'features.13', 'features.16', 'features.18',
                'classifier.0', 'classifier.3']


class VGG11BN(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG11BN, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['A'], batch_norm=True),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG11BN, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.1', 'features.5', 'features.9', 'features.12',
                'features.16', 'features.19', 'features.23', 'features.26',
                'classifier.0', 'classifier.3']'''


class VGG16(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG16, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['D'], batch_norm=False),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG16, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.0', 'features.2',
                'features.5', 'features.7',
                'features.10', 'features.12', 'features.14',
                'features.17', 'features.19', 'features.21',
                'features.24', 'features.26', 'features.28',
                'classifier.0', 'classifier.3']


class VGG16BN(torchvision.models.VGG):
    def __init__(self, num_classes, use_dropout=False, **kwargs):
        super(VGG16BN, self).__init__(
            torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['D'], batch_norm=True),
            num_classes,
            **kwargs
        )
        if not use_dropout:
            self.classifier[2] = nn.Identity()
            self.classifier[5] = nn.Identity()

    def forward(self, x):
        return super(VGG16BN, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['features.1', 'features.4',
                'features.8', 'features.11',
                'features.15', 'features.18', 'features.21',
                'features.25', 'features.28', 'features.31',
                'features.35', 'features.38', 'features.41',
                'classifier.0', 'classifier.3']


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
                'classifier.0', 'classifier.3']


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
                'classifier.0', 'classifier.3']


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

    # TODO: Add block outputs to relevant layers!
    @staticmethod
    def get_relevant_layers():
        return ['bn1',
                'layer1.0.bn1', 'layer1.0.bn2', 'layer1.1.bn1', 'layer1.1.bn2',
                'layer2.0.bn1', 'layer2.0.bn2', 'layer2.1.bn1', 'layer2.1.bn2',
                'layer3.0.bn1', 'layer3.0.bn2', 'layer3.1.bn1', 'layer3.1.bn2',
                'layer4.0.bn1', 'layer4.0.bn2', 'layer4.1.bn1', 'layer4.1.bn2']


class ResNet34(torchvision.models.ResNet):
    def __init__(self, num_classes, **kwargs):
        super(ResNet34, self).__init__(
            torchvision.models.resnet.BasicBlock,
            [3, 4, 6, 3],
            num_classes,
            **kwargs
        )

    def forward(self, x):
        return super(ResNet34, self).forward(x)

    # TODO: Add block outputs to relevant layers!
    @staticmethod
    def get_relevant_layers():
        return ['bn1',
                'layer1.0.bn1', 'layer1.1.bn1', 'layer1.2.bn1',
                'layer2.0.bn1', 'layer2.1.bn1', 'layer2.2.bn1', 'layer2.3.bn1',
                'layer3.0.bn1', 'layer3.1.bn1', 'layer3.2.bn1', 'layer3.3.bn1', 'layer3.4.bn1', 'layer3.5.bn1',
                'layer4.0.bn1', 'layer4.1.bn1', 'layer4.2.bn1']


class ResNet50(torchvision.models.ResNet):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__(
            torchvision.models.resnet.Bottleneck,
            [3, 4, 6, 3],
            num_classes,
            **kwargs
        )

    def forward(self, x):
        return super(ResNet50, self).forward(x)

    # TODO: Add block outputs to relevant layers!
    @staticmethod
    def get_relevant_layers():
        return ['bn1',
                'layer1.0.bn3', 'layer1.1.bn3', 'layer1.2.bn3',
                'layer2.0.bn3', 'layer2.1.bn3', 'layer2.2.bn3', 'layer2.3.bn3',
                'layer3.0.bn3', 'layer3.1.bn3', 'layer3.2.bn3', 'layer3.3.bn3', 'layer3.4.bn3',
                'layer4.0.bn3', 'layer4.1.bn3', 'layer4.2.bn3']
