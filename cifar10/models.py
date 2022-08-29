# Original code from: https://github.com/hongyi-zhang/Fixup
import torch
from torch import nn
import numpy as np
from torchvision.transforms.functional import normalize as standardize


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = standardize(x, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x + self.bias2)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = standardize(x, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class FixupResNet110(FixupResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(FixupBasicBlock, [18, 18, 18], *args, **kwargs)

    @staticmethod
    def get_relevant_layers():
        return ['conv1',
                'layer1.0.conv1', 'layer1.0', 'layer1.1.conv1', 'layer1.1', 'layer1.2.conv1', 'layer1.2',
                'layer1.3.conv1', 'layer1.3', 'layer1.4.conv1', 'layer1.4', 'layer1.5.conv1', 'layer1.5',
                'layer1.6.conv1', 'layer1.6', 'layer1.7.conv1', 'layer1.7', 'layer1.8.conv1', 'layer1.8',
                'layer1.9.conv1', 'layer1.9', 'layer1.10.conv1', 'layer1.10', 'layer1.11.conv1', 'layer1.11',
                'layer1.12.conv1', 'layer1.12', 'layer1.13.conv1', 'layer1.13', 'layer1.14.conv1', 'layer1.14',
                'layer1.15.conv1', 'layer1.15', 'layer1.16.conv1', 'layer1.16', 'layer1.17.conv1', 'layer1.17',

                'layer2.0.conv1', 'layer2.0', 'layer2.1.conv1', 'layer2.1', 'layer2.2.conv1', 'layer2.2',
                'layer2.3.conv1', 'layer2.3', 'layer2.4.conv1', 'layer2.4', 'layer2.5.conv1', 'layer2.5',
                'layer2.6.conv1', 'layer2.6', 'layer2.7.conv1', 'layer2.7', 'layer2.8.conv1', 'layer2.8',
                'layer2.9.conv1', 'layer2.9', 'layer2.10.conv1', 'layer2.10', 'layer2.11.conv1', 'layer2.11',
                'layer2.12.conv1', 'layer2.12', 'layer2.13.conv1', 'layer2.13', 'layer2.14.conv1', 'layer2.14',
                'layer2.15.conv1', 'layer2.15', 'layer2.16.conv1', 'layer2.16', 'layer2.17.conv1', 'layer2.17',

                'layer3.0.conv1', 'layer3.0', 'layer3.1.conv1', 'layer3.1', 'layer3.2.conv1', 'layer3.2',
                'layer3.3.conv1', 'layer3.3', 'layer3.4.conv1', 'layer3.4', 'layer3.5.conv1', 'layer3.5',
                'layer3.6.conv1', 'layer3.6', 'layer3.7.conv1', 'layer3.7', 'layer3.8.conv1', 'layer3.8',
                'layer3.9.conv1', 'layer3.9', 'layer3.10.conv1', 'layer3.10', 'layer3.11.conv1', 'layer3.11',
                'layer3.12.conv1', 'layer3.12', 'layer3.13.conv1', 'layer3.13', 'layer3.14.conv1', 'layer3.14',
                'layer3.15.conv1', 'layer3.15', 'layer3.16.conv1', 'layer3.16', 'layer3.17.conv1', 'layer3.17']


class ResNet110(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(BasicBlock, [18, 18, 18], *args, **kwargs)

    @staticmethod
    def get_relevant_layers():
        return ['conv1',
                'layer1.0.bn1', 'layer1.0', 'layer1.1.bn1', 'layer1.1', 'layer1.2.bn1', 'layer1.2',
                'layer1.3.bn1', 'layer1.3', 'layer1.4.bn1', 'layer1.4', 'layer1.5.bn1', 'layer1.5',
                'layer1.6.bn1', 'layer1.6', 'layer1.7.bn1', 'layer1.7', 'layer1.8.bn1', 'layer1.8',
                'layer1.9.bn1', 'layer1.9', 'layer1.10.bn1', 'layer1.10', 'layer1.11.bn1', 'layer1.11',
                'layer1.12.bn1', 'layer1.12', 'layer1.13.bn1', 'layer1.13', 'layer1.14.bn1', 'layer1.14',
                'layer1.15.bn1', 'layer1.15', 'layer1.16.bn1', 'layer1.16', 'layer1.17.bn1', 'layer1.17',

                'layer2.0.bn1', 'layer2.0', 'layer2.1.bn1', 'layer2.1', 'layer2.2.bn1', 'layer2.2',
                'layer2.3.bn1', 'layer2.3', 'layer2.4.bn1', 'layer2.4', 'layer2.5.bn1', 'layer2.5',
                'layer2.6.bn1', 'layer2.6', 'layer2.7.bn1', 'layer2.7', 'layer2.8.bn1', 'layer2.8',
                'layer2.9.bn1', 'layer2.9', 'layer2.10.bn1', 'layer2.10', 'layer2.11.bn1', 'layer2.11',
                'layer2.12.bn1', 'layer2.12', 'layer2.13.bn1', 'layer2.13', 'layer2.14.bn1', 'layer2.14',
                'layer2.15.bn1', 'layer2.15', 'layer2.16.bn1', 'layer2.16', 'layer2.17.bn1', 'layer2.17',

                'layer3.0.bn1', 'layer3.0', 'layer3.1.bn1', 'layer3.1', 'layer3.2.bn1', 'layer3.2',
                'layer3.3.bn1', 'layer3.3', 'layer3.4.bn1', 'layer3.4', 'layer3.5.bn1', 'layer3.5',
                'layer3.6.bn1', 'layer3.6', 'layer3.7.bn1', 'layer3.7', 'layer3.8.bn1', 'layer3.8',
                'layer3.9.bn1', 'layer3.9', 'layer3.10.bn1', 'layer3.10', 'layer3.11.bn1', 'layer3.11',
                'layer3.12.bn1', 'layer3.12', 'layer3.13.bn1', 'layer3.13', 'layer3.14.bn1', 'layer3.14',
                'layer3.15.bn1', 'layer3.15', 'layer3.16.bn1', 'layer3.16', 'layer3.17.bn1', 'layer3.17']
