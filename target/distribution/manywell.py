from typing import Callable

import numpy as np
import torch
import torch.distributions as D

from .base_set import BaseSet
from target.plot import traj_plot1d, dist_plot1d, viz_sample2d, viz_kde2d, viz_contour_sample2d


def rejection_sampling(n_samples: int, proposal: torch.distributions.Distribution,
                       target_log_prob_fn: Callable, k: float) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample((n_samples*10,))
    u_0 = torch.distributions.Uniform(0, k*torch.exp(proposal.log_prob(z_0)))\
        .sample().to(z_0)
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_log_prob_fn, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples

class ManyWell(BaseSet):
    """
    log p(x1, x2) = −x1^4 + 6*x1^2 + 1/2*x1 − 1/2*x2^2 + constant
    """
    def __init__(self, len_data, dim=32, is_linear=True):
        super().__init__(len_data)
        self.data = torch.ones(dim, dtype=float).cuda()
        self.data_ndim = dim
        assert dim % 2 == 0
        self.n_wells = dim // 2

        # as rejection sampling proposal
        self.component_mix = torch.tensor([0.2, 0.8])
        self.means = torch.tensor([-1.7, 1.7])
        self.scales = torch.tensor([0.5, 0.5])

        self.Z_x1 = 11784.50927
        self.logZ_x2 = 0.5*np.log(2*np.pi)
        self.logZ_doublewell = np.log(self.Z_x1) + self.logZ_x2

    def gt_logz(self):
        return self.n_wells * self.logZ_doublewell

    def energy(self, x):
        return -self.manywell_logprob(x)

    def doublewell_logprob(self, x):
        assert x.shape[1] == 2 and x.ndim == 2
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_term = 0.5*x1 + 6*x1.pow(2) - x1.pow(4)
        x2_term = -0.5*x2.pow(2)
        return x1_term + x2_term

    def manywell_logprob(self, x):
        assert x.ndim == 2
        logprob = torch.stack(
            [self.doublewell_logprob(x[:, i*2:i*2+2]) for i in range(self.n_wells)],
        dim=1).sum(dim=1)
        return logprob

    def sample_first_dimension(self, batch_size):
        def target_log_prob(x):
            return -x ** 4 + 6 * x ** 2 + 1 / 2 * x

        # Define proposal
        mix = torch.distributions.Categorical(self.component_mix)
        com = torch.distributions.Normal(self.means, self.scales)
        proposal = torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                         component_distribution=com)

        k = self.Z_x1 * 3
        samples = rejection_sampling(batch_size, proposal, target_log_prob, k)
        return samples

    def sample_doublewell(self, batch_size):
        x1 = self.sample_first_dimension(batch_size)
        x2 = torch.randn_like(x1)
        return torch.stack([x1, x2], dim=1)

    def sample(self, batch_size):
        return torch.cat(
            [self.sample_doublewell(batch_size) for _ in range(self.n_wells)],
        dim=-1)

    def viz_pdf(self, samples=None, num_samples=5000):
        if samples is None:
            samples = self.sample(num_samples)

        x13 = samples[:, 0:3:2]
        viz_sample2d(x13, "samples", f"distx13.png", lim=3)
        viz_kde2d(x13, "kde", f"kdex13.png", lim=3)

        lim = 3
        alpha = 0.8
        n_contour_levels = 20
        def logp_func(x_2d):
            x = torch.zeros((x_2d.shape[0], self.data_ndim))
            x[:, 0] = x_2d[:, 0]
            x[:, 2] = x_2d[:, 1]
            return -self.energy(x)
        contour_img_path = f"contourx13.png"
        viz_contour_sample2d(x13, contour_img_path, logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)

        x23 = samples[:, 1:3]
        viz_sample2d(x23, "samples", f"distx23.png", lim=3)
        viz_kde2d(x23, "kde", f"kdex23.png", lim=3)

        def logp_func(x_2d):
            x = torch.zeros((x_2d.shape[0], self.data_ndim))
            x[:, 1] = x_2d[:, 0]
            x[:, 2] = x_2d[:, 1]
            return -self.energy(x)
        contour_img_path2 = f"contourx23.png"
        viz_contour_sample2d(x23, contour_img_path2, logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)
