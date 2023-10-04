import numpy as np
import torch


def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / (sigma ** 2)
    return torch.sum(torch.flatten(loss, start_dim=1), -1)

def linear_intepolate_energy(origin_energy_fn, x, weight_gauss=1.0):
    origin_energy = origin_energy_fn(x)
    gaussian_energy = nll_unit_gaussian(x)
    assert origin_energy.shape == gaussian_energy.shape
    return weight_gauss * gaussian_energy + (1.0 - weight_gauss) * origin_energy