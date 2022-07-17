import torch


def project(x, original_x, epsilon):
    max_x = original_x + epsilon
    min_x = original_x - epsilon

    x = torch.max(torch.min(x, max_x), min_x)

    return x


class LinfProjectedGradientDescendAttack:
    def __init__(self, model, loss_fn, eps, step_size, steps, random_start=True, reg=lambda: 0.0, bounds=(0.0, 1.0), device=None):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.bounds = bounds
        self.steps = steps

        self.random_start = random_start

        self.reg = reg

        self.device = device if device else torch.device('cpu')

    '''def perturb(self, original_x, labels, random_start=True):
        model_original_mode = self.model.training
        self.model.eval()
        if random_start:
            rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
            rand_perturb = rand_perturb.to(self.device)
            x = original_x + rand_perturb
            x.clamp_(self.bounds[0], self.bounds[1])
        else:
            x = original_x.clone()

        x.requires_grad = True

        with torch.enable_grad():
            for _iter in range(self.steps):
                outputs = self.model(x)

                loss = self.loss_fn(outputs, labels) + self.reg()

                grads = torch.autograd.grad(loss, x)[0]

                x.data += self.step_size * torch.sign(grads.data)

                x = project(x, original_x, self.eps)
                x.clamp_(self.bounds[0], self.bounds[1])

        self.model.train(mode=model_original_mode)
        return x'''

    def perturb(self, original_x, y):
        if self.random_start:
            rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
            rand_perturb = rand_perturb.to(self.device)
            x = original_x.detach() + rand_perturb
            x.clamp_(self.bounds[0], self.bounds[1])
        else:
            x = original_x.detach()

        for _iter in range(self.steps):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y) + self.reg()
            grads = torch.autograd.grad(loss, x)[0]
            x = x.detach() + self.step_size * torch.sign(grads.detach())
            x = project(x, original_x, self.eps)
            x.clamp_(self.bounds[0], self.bounds[1])
        return x

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


LinfPGDAttack = LinfProjectedGradientDescendAttack


parameter_presets = {
    'MNIST': {'eps': 0.3, 'step_size': 0.01, 'steps': 40},
    'FashionMNIST': {'eps': 0.1, 'step_size': 0.01, 'steps': 40},
    'CIFAR10': {'eps': 8 / 255, 'step_size': 2 / 255, 'steps': 10}
}
