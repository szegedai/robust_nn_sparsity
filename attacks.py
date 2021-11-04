import torch


def project(x, original_x, epsilon):
    max_x = original_x + epsilon
    min_x = original_x - epsilon

    x = torch.max(torch.min(x, max_x), min_x)

    return x


class LinfProjectedGradientDescendAttack:
    def __init__(self, model, loss_fn, eps, step_size, steps, bounds=(0.0, 1.0), device=None):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.bounds = bounds
        self.steps = steps

        self.device = device if device else torch.device('cpu')

    def __call__(self, original_images, labels, random_start=False):
        model_original_mode = self.model.training
        self.model.training = False
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(-self.eps, self.eps)
            rand_perturb = rand_perturb.to(self.device)
            x = original_images + rand_perturb
            x.clamp_(self.bounds[0], self.bounds[1])
        else:
            x = original_images.clone()

        x.requires_grad = True

        with torch.enable_grad():
            for _iter in range(self.steps):
                outputs = self.model(x)

                loss = self.loss_fn(outputs, labels)

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True)[0]

                x.data += self.step_size * torch.sign(grads.data)

                x = project(x, original_images, self.eps)
                x.clamp_(self.bounds[0], self.bounds[1])

        self.model.training = model_original_mode
        return x


LinfPGDAttack = LinfProjectedGradientDescendAttack