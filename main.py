import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from attacks import LinfPGDAttack
from training import *
from activations import *
from models import *
from utils import MultiDataset, split_dataset, MerticsLogManager


def activation_dif_experiment():
    pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_split = 0.1

    '''model = VGG4((1, 28, 28), 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    attack = LinfPGDAttack(model, loss_fn, eps=0.3, step_size=0.01, steps=40, device=device)'''

    train_dataset = torchvision.datasets.MNIST('./datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=6)
    test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=6)
    train_dataset, val_dataset = split_dataset(train_dataset, val_split)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=6)

    combined_dataset = MultiDataset(train_dataset, val_dataset, test_dataset)
    combined_loader = DataLoader(combined_dataset, batch_size=50, shuffle=True, num_workers=6)

    loss_fn = nn.CrossEntropyLoss()
    for sw in range(10, 41, 10):
        model = VGG4((1, 28, 28), 10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        attack = LinfPGDAttack(model, loss_fn, eps=0.3, step_size=0.01, steps=40, device=device)

        train_static_hybrid(model, loss_fn, optimizer, attack, train_loader, val_loader,
                            switch_point=sw, num_epochs=50,
                            checkpoint_dir=f'hybrid/hybrid_sw{sw}',
                            metrics_lm=MerticsLogManager(f'hybrid/hybrid_sw{sw}.log'))

    '''train_adv(model, loss_fn, optimizer, attack, train_loader, num_epochs=5)
    activations = get_activations(model, ['conv2d_0', 'conv2d_1', 'linear_0'], combined_loader)
    print(get_inactivity_ratio(activations))'''

    '''model = torchvision.models.resnet18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    attack = LinfPGDAttack(model, loss_fn, eps=2/255, step_size=8/255, steps=10, device=device)

    train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)'''

    #train_adv(model, loss_fn, optimizer, attack, train_loader, num_epochs=20)
    #train_dynamic_hybrid(model, loss_fn, optimizer, attack, train_loader, loss_window=5, loss_deviation=0.05, num_epochs=50)
    #train_static_hybrid(model, loss_fn, optimizer, attack, train_loader, switch_point=10, num_epochs=12)


if __name__ == '__main__':
    main()
