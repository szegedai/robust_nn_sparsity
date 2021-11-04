import torch
import torch.nn as nn


class ActivationExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers=None):
        super().__init__()
        self.model = model
        if layers is None:
            self.layers = []
            for n, _ in model.named_modules():
                self.layers.append(n)
        else:
            self.layers = layers
        self.activations = {layer: torch.empty(0) for layer in self.layers}

        self.hooks = []

        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))

    def get_activation_hook(self, layer_id: str):
        def fn(_, __, output):
            self.activations[layer_id] = output

        return fn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        self.model(x)
        return self.activations


def get_activations(model: nn.Module, layers, dataloader):
    with torch.no_grad():
        model_device = next(model.parameters()).device
        extractor = ActivationExtractor(model, layers)
        it = iter(dataloader)

        max_activations = {}
        activations = extractor(next(it)[0].to(model_device))
        for k, v in activations.items():
            max_activations[k] = torch.max(v, dim=0)[0]

        while (batch := next(it, None)) is not None:
            activations = extractor(batch[0].to(model_device))
            for k, v in activations.items():
                max_activations[k] = torch.maximum(torch.max(v, dim=0)[0], max_activations[k])
        extractor.remove_hooks()
    return max_activations


def get_inactivity_ratio(activations):
    ratios = {}
    for k, v in activations.items():
        ratios[k] = ((torch.numel(v) - torch.count_nonzero(v)) / torch.numel(v)).item()
    return ratios
