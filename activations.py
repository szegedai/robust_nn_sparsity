import torch
import torch.nn as nn


'''def split_dict(d: dict):
    return d.keys(), torch.stack(list(d.values()))


def create_dict(k, v: torch.Tensor):
    return dict(zip(k, torch.unbind(v)))'''


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


def get_activations(model: nn.Module, layers, dataloader, attack=None):
    with torch.no_grad():
        model_device = next(model.parameters()).device
        extractor = ActivationExtractor(model, layers)
        it = iter(dataloader)

        max_activations = {}
        batch = next(it)
        batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
        if attack is not None:
            activations = extractor(attack(*batch))
        else:
            activations = extractor(batch[0])
        for k, v in activations.items():
            max_activations[k] = torch.max(v, dim=0)[0]

        while (batch := next(it, None)) is not None:
            batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
            if attack is not None:
                activations = extractor(attack(*batch))
            else:
                activations = extractor(batch[0])
            for k, v in activations.items():
                max_activations[k] = torch.maximum(torch.max(v, dim=0)[0], max_activations[k])
        extractor.remove_hooks()
    return max_activations


def get_activation_counts(model, layers, dataloader, attack=None):
    with torch.no_grad():
        model_device = next(model.parameters()).device
        extractor = ActivationExtractor(model, layers)
        it = iter(dataloader)

        num_samples = 0
        activation_counts = {}
        batch = next(it)
        batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
        num_samples += len(batch[0])
        if attack is not None:
            activations = extractor(attack(*batch))
        else:
            activations = extractor(batch[0])
        for k, v in activations.items():
            activation_counts[k] = torch.sum(torch.sign(v), dim=0)

        while (batch := next(it, None)) is not None:
            batch[0], batch[1] = batch[0].to(model_device), batch[1].to(model_device)
            num_samples += len(batch[0])
            if attack is not None:
                activations = extractor(attack(*batch))
            else:
                activations = extractor(batch[0])
            for k, v in activations.items():
                activation_counts[k] += torch.sum(torch.sign(v), dim=0)
        extractor.remove_hooks()
    return activation_counts, num_samples


def get_activity(model, layers, dataloader, attack=None):
    activation_counts, num_samples = get_activation_counts(model, layers, dataloader, attack)
    with torch.no_grad():
        for k, v in activation_counts.items():
            activation_counts[k] /= num_samples  # turn count into ratios
    return activation_counts


def get_inactivity(model, layers, dataloader, attack=None):
    activation_counts, num_samples = get_activation_counts(model, layers, dataloader, attack)
    with torch.no_grad():
        for k, v in activation_counts.items():
            activation_counts[k] /= num_samples  # turn count into ratios
            activation_counts[k] = 1 - activation_counts[k]
    return activation_counts


def get_inactivity_ratios(activations):  # works with activation dicts and activity dicts too
    ratios = {}
    for k, v in activations.items():
        ratios[k] = ((torch.numel(v) - torch.count_nonzero(v)) / torch.numel(v)).item()
    return ratios


def get_inactivity_ratio(activations):  # works with activation dicts and activity dicts too
    num_activations = 0.0
    num_inactives = 0.0
    for k, v in activations.items():
        num_activations += torch.numel(v)
        num_inactives += (torch.numel(v) - torch.count_nonzero(v)).item()
    return num_inactives / num_activations


# WIP!
def reinitialize_inactive_neuron_weights(model: nn.Module, activations: dict, layers=None, ration=1.0):
    if layers is None:
        layers = activations.keys()
    device = next(model.parameters()).device
    for k in layers:
        layer = model.get_submodule(k)
        activation = activations[k]
        w_std_mean = torch.std_mean(layer.weight.detach())
        b_std_mean = torch.std_mean(layer.bias.detach())
        mask = torch.all((activation == 0).view(activation.shape[0], -1), dim=-1)
        with torch.no_grad():
            layer.weight.masked_scatter_(
                torch.repeat_interleave(mask, torch.tensor(layer.weight.shape[1:], device=device).prod()).view(layer.weight.shape),
                torch.normal(w_std_mean[1].item(), w_std_mean[0].item(), layer.weight.shape, device=device)
            )
            layer.bias.masked_scatter_(
                mask,
                torch.normal(b_std_mean[1].item(), b_std_mean[0].item(), layer.bias.shape, device=device)
            )
