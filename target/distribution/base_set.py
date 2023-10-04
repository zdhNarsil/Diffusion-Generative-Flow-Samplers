import abc
import torch
from torch.utils.data import Dataset

from target.ksd import KSD
from target.loss_helper import linear_intepolate_energy, nll_unit_gaussian


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None
        self.worker = KSD(self.score, beta=0.2)
        self._gt_ksd = None

    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x):
        return

    def unnorm_pdf(self, x):
        return torch.exp(-self.energy(x))

    # hmt stands for hamiltonian
    def hmt_energy(self, x):
        dim = x.shape[-1]
        x, v = torch.split(x, dim // 2, dim=-1)
        neg_log_p_x = self.sample_energy_fn(x)
        neg_log_p_v = nll_unit_gaussian(v)
        return neg_log_p_x + neg_log_p_v

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def hmt_score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.hmt_energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def ksd(self, points):
        with torch.no_grad():
            cur_ksd = self.gt_ksd()
        return self.worker(points) - cur_ksd

    def gt_ksd(self):
        if self._gt_ksd is None:
            with torch.no_grad():
                self._gt_ksd = self.worker(
                    self.sample(5000).view(5000, -1), adjust_beta=True
                )
        return self._gt_ksd
