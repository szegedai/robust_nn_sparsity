import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from attacks import LinfPGDAttack
from training import *
from activations import *
from models import *
from utils import MultiDataset


def activation_dif_experiment():
    pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VGG4((1, 28, 28), 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    attack = LinfPGDAttack(model, loss_fn, eps=0.3, step_size=0.01, steps=40, device=device)

    train_dataset = torchvision.datasets.MNIST('./datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
    test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=8)

    combined_dataset = MultiDataset(train_dataset, test_dataset)
    combined_loader = DataLoader(combined_dataset, batch_size=50, shuffle=True, num_workers=8)

    relevant_layers = ['conv2d_0', 'conv2d_1', 'linear_0', 'linear_1']
    activations = get_activations(model, relevant_layers, combined_loader)
    print(get_inactivity_ratio(activations))
    train_adv(model, loss_fn, optimizer, train_loader, attack, num_epochs=10, device=device)
    activations = get_activations(model, relevant_layers, combined_loader)
    print(get_inactivity_ratio(activations))


    '''model = torchvision.models.resnet18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    attack = LinfPGDAttack(model, loss_fn, eps=2/255, step_size=8/255, steps=10, device=device)

    train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    #train_adv(model, loss_fn, optimizer, train_loader, attack, num_epochs=20, device=device)
    #train_dynamic_hybrid(model, loss_fn, optimizer, train_loader, attack, loss_window=5, loss_deviation=0.05, num_epochs=100, device=device)
    #train_static_hybrid(model, loss_fn, optimizer, train_loader, attack, switch_point=35, num_epochs=50, device=device)'''


if __name__ == '__main__':
    main()
