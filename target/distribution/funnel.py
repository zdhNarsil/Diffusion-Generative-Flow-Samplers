import numpy as np
import torch
import torch as th
import torch.distributions as D

from .base_set import BaseSet


class FunnelSet(BaseSet):
    """
    x0 ~ N(0, 3^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
    """
    def __init__(self, dim):
        super().__init__()
        self.data = th.ones(dim, dtype=float).cuda()
        self.data_ndim = dim

        self.dist_dominant = D.Normal(th.tensor([0.0]).cuda(), th.tensor([3.0]).cuda())
        self.mean_other = th.zeros(dim - 1).float().cuda()
        self.cov_eye = th.eye(dim - 1).float().cuda().view(1, dim - 1, dim - 1)

    # def cal_gt_big_z(self):
    #     return 1

    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.funner_log_pdf(x)

    def viz_pdf(self, fsave="density.png", lim=3):
        raise NotImplementedError

    def funner_log_pdf(self, x):
        try:
            dominant_x = x[:, 0]
            log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )
            # log_density_other = self._dist_other(dominant_x).log_prob(x[:, 1:])  # (B, )

            log_sigma = 0.5 * x[:, 0:1]
            sigma2 = torch.exp(x[:, 0:1])
            neglog_density_other = 0.5*np.log(2*np.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
            log_density_other = torch.sum(-neglog_density_other, dim=-1)
        except:
            import ipdb;
            ipdb.set_trace()
        return log_density_dominant + log_density_other

    def sample(self, batch_size):
        dominant_x = self.dist_dominant.sample((batch_size,))  # (B,1)
        x_others = self._dist_other(dominant_x).sample()  # (B, dim-1)
        return th.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = th.exp(dominant_x)
        cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return D.multivariate_normal.MultivariateNormal(self.mean_other, cov_other)