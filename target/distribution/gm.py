import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet

"""
2-D Guassian mixture
"""
class GaussianMixture2D(BaseSet):
    def __init__(self, scale=0.5477222, # sqrt(0.3)
                 # nmode=9, xlim=1.0
                 ):
        super().__init__()
        # xlim = 0.01 if nmode == 1 else xlim
        self.data = torch.tensor([0.0])

        mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]
        nmode = len(mean_ls)
        mean = torch.stack([torch.tensor(xy) for xy in mean_ls])
        comp = D.Independent(D.Normal(mean.cuda(), torch.ones_like(mean).cuda() * scale), 1)
        mix = D.Categorical(torch.ones(nmode).cuda())
        self.gmm = MixtureSameFamily(mix, comp)

    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.gmm.log_prob(x).flatten()

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))

    def viz_pdf(self, fsave="ou-density.png"):
        x = torch.linspace(-8, 8, 100).cuda()
        y = torch.linspace(-8, 8, 100).cuda()
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1) #?

        density = self.unnorm_pdf(x)
        # x, pdf = as_numpy([x, density])
        x, pdf = torch.from_numpy(x), torch.from_numpy(density)

        fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
        axs.plot(x, pdf)

        # plt.contourf(X, Y, density, levels=20, cmap='viridis')
        # plt.colorbar()
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('2D Function Plot')

        fig.savefig(fsave)
        plt.close(fig)

    def __getitem__(self, idx):
        del idx
        return self.data[0]
