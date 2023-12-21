import copy
from time import time
from hydra.utils import instantiate

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from gflownet.network import FourierMLP, TimeConder
from gflownet.utils import normal_logp, num_to_groups, loss2ess_info
from target.plot import viz_sample2d, viz_kde2d, viz_contour_sample2d


def get_alg(cfg, task=None):
    # alg = SubTrajectoryBalanceTransitionBased(cfg, task=task)
    alg = SubTrajectoryBalanceTrajectoryBased(cfg, task=task)
    return alg

def fl_inter_logr(x, logreward_fn, config, cur_t, sigma=None): # x: (bs, dim)
    if sigma is None:
        sigma = config.sigma

    if isinstance(cur_t, torch.Tensor):
        # print(cur_t.shape) # (bs, 1) or (,)
        if cur_t.ndim <= 1:
            cur_t = cur_t.item()
    ratio = cur_t / config.t_end
    # assert 0 <= ratio <= 1

    coef = max(np.sqrt(0.01 * config.t_end), np.sqrt(cur_t))  # cur_t could be 0
    logp0 = normal_logp(x, 0., coef * sigma)
    fl_logr = logreward_fn(x) * ratio + logp0 * (1 - ratio)

    return fl_logr

def sample_traj(gfn, config, logreward_fn, batch_size=None, sigma=None):
    if batch_size is None:
        batch_size = config.batch_size
    device = gfn.device
    if sigma is None:
        sigma = config.sigma

    x = gfn.zero(batch_size).to(device)
    fl_logr = fl_inter_logr(x, logreward_fn, config, cur_t=0., sigma=sigma)
    traj = [(torch.tensor(0.), x.cpu(), fl_logr.cpu())] # (t, x, logr)
    inter_loss = torch.zeros(batch_size).to(device)

    x_max = 0.
    for cur_t in torch.arange(0, config.t_end, config.dt).to(device):
        x, uw_term, u2_term = gfn.step_forward(cur_t, x, config.dt, sigma=sigma)
        x = x.detach()
        fl_logr = fl_inter_logr(x, logreward_fn, config,
            cur_t=cur_t + config.dt, sigma=sigma).detach().cpu()
        traj.append((cur_t.cpu() + config.dt, x.detach().cpu(), fl_logr))
        inter_loss += (u2_term + uw_term).detach()
        x_max = max(x_max, x.abs().max().item())

    pis_terminal = -gfn.nll_prior(x) - logreward_fn(x)
    pis_log_weight = inter_loss + pis_terminal
    info = {"pis_logw": pis_log_weight, "x_max": x_max}
    return traj, info


class GFlowNet(nn.Module):
    """
    For PIS modeling: s0 is fixed to be zero
    thus PB(s0|s1) == 1
    """
    def __init__(self, cfg, task=None):
        super().__init__()
        self.cfg = cfg

        self.data_ndim = cfg.data_ndim  # int(np.prod(data_shape))
        self.register_buffer("x0", torch.zeros((1, self.data_ndim))) # for pis modeling
        self.t_end = cfg.t_end
        self.dt = cfg.dt

        self.g_func = instantiate(cfg.g_func)
        self.f_func = instantiate(cfg.f_func) # learnable
        self.nn_clip = cfg.nn_clip
        self.lgv_clip = cfg.lgv_clip
        self.task = task
        self.grad_fn = task.score
        self.select_f(f_format=cfg.f)
        self.xclip = cfg.xclip
        self.logr_fn = lambda x: -task.energy(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def sigma(self): # simply return a float w/o grad
        return self.cfg.sigma

    def save(self, path):
        raise NotImplementedError

    def get_optimizer(self):
        return torch.optim.Adam(self.param_ls, weight_decay=self.cfg.weight_decay)

    def select_f(self, f_format=None):
        _fn = self.f_func
        if f_format == "f":
            def _fn(t, x):
                return torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
        elif f_format in ["t_tnet_grad", "tgrad"]:
            self.lgv_coef = TimeConder(self.cfg.f_func.channels, 1, 3)
            def _fn(t, x):
                grad = torch.nan_to_num(self.grad_fn(x))
                grad = torch.clip(grad, -self.lgv_clip, self.lgv_clip)

                f = torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                lgv_coef = self.lgv_coef(t)
                return f - lgv_coef * grad
        else:
            raise RuntimeError
        self.param_fn = _fn

    # t: scalar; state: (b, state_dim)
    def f(self, t, state): # same as self.param_fn
        x = torch.nan_to_num(state)
        control = self.param_fn(t, x) #.flatten(start_dim=1)
        return control

    def zero(self, batch_size, device=None):
        device = self.device if device is None else device
        return self.x0.expand(batch_size, -1).to(device)

    def nll_prior(self, state): # nll on terminal state
        return -normal_logp(state, 0., np.sqrt(self.t_end) * self.sigma)

    # t -> t + dt; n -> n + 1; here t is scalar tensor
    def step_forward(self, t, state, dt, sigma=None, return_drift_scale=False):
        sigma = self.sigma if sigma is None else sigma
        std_noise = self.g_func(t, state) * torch.randn_like(state)

        noise_term = std_noise * sigma * np.sqrt(dt)
        pre_drift = self.f(t, state)
        # drift_scale = pre_drift.norm(dim=-1).mean()
        next_state = state + pre_drift * sigma * dt + noise_term
        u2_term = 0.5 * pre_drift.pow(2).sum(dim=-1) * dt
        uw_term = (pre_drift * std_noise).sum(dim=1) * np.sqrt(dt)

        if self.xclip > 0: # avoid nan
            next_state = torch.clip(next_state, -self.xclip, self.xclip)

        return next_state, uw_term, u2_term

    # t -> t - dt; n -> n - 1
    def step_backward(self, t ,state, dt): # not used
        std_noise = self.g_func(t, state) * torch.randn_like(state)
        # n = (t - dt) / dt
        coef = (t-dt)/t  # = n/(n + 1)
        mean = self.x0*dt / t + coef * state
        noise_term = std_noise * self.sigma * np.sqrt(dt)
        prev_state = mean + coef.sqrt() * noise_term
        return prev_state.detach()

    def log_pf(self, t, state, next_state, dt=None): # t: (bs, 1), dt: float
        assert t.ndim == state.ndim == next_state.ndim == 2
        dt = self.dt if dt is None else dt
        sigma = self.cfg.sigma
        mean = state + self.f(t, state) * sigma * dt
        log_pf = normal_logp(next_state, mean, sigma * np.sqrt(dt))
        return log_pf

    def log_pb(self, t, state, prev_state, dt=None): # t: (bs, 1), dt: float
        assert t.ndim == state.ndim == prev_state.ndim == 2
        dt = self.dt if dt is None else dt
        sigma = self.cfg.sigma

        mean = self.x0 * dt / t + (t - dt) / t * state
        sigma_pb = ((t-dt)/t).sqrt() * np.sqrt(dt) * sigma
        log_pb = normal_logp(prev_state, mean, sigma_pb)

        # first step is from Dirac on x0 to a Gaussian, thus PB == 1
        # sigma_pb.min = 0 => nan
        first_step_mask = (t <= dt).squeeze(dim=1)
        # log_pb[first_step_mask] = 0.
        log_pb = torch.where(first_step_mask, torch.zeros_like(log_pb), log_pb)
        return log_pb

    def log_pf_and_pb_traj(self, traj):
        batch_size = traj[0][1].shape[0]

        xs = [x for (t, x, r) in traj]
        state = torch.cat(xs[:-1], dim=0).to(self.device)  # (N*b, d)
        next_state = torch.cat(xs[1:], dim=0).to(self.device)  # (N*b, d)
        ts = [repeat(t[None], "one -> b one", b=batch_size) for (t, x, r) in traj]
        time = torch.cat(ts[:-1], dim=0).to(self.device)  # (N*b, 1)
        next_time = torch.cat(ts[1:], dim=0).to(self.device)  # (N*b, 1)

        log_pb = self.log_pb(next_time, next_state, state)
        log_pb = log_pb.detach()

        if self.cfg.task in ["cox"]:  # save cuda memory
            start_idx = 0
            log_pf = torch.zeros((0,)).to(self.device)
            for bs in num_to_groups(log_pb.shape[0], 1000):
                log_pf_curr = self.log_pf(time[start_idx:start_idx+bs],
                          state[start_idx:start_idx+bs], next_state[start_idx:start_idx+bs])
                log_pf = torch.cat([log_pf, log_pf_curr], dim=0)
                start_idx += bs
        else:
            log_pf = self.log_pf(time, state, next_state)

        return log_pf, log_pb

    def log_weight(self, traj): # "log q - log p" in VAE notation
        batch_size = traj[0][1].shape[0]
        logr = self.logr_from_traj(traj)
        log_pf, log_pb = self.log_pf_and_pb_traj(traj) # (N*b,)
        dlogp = reduce(log_pf - log_pb, "(N b) -> b", "sum", b=batch_size)
        return dlogp - logr

    def logr_from_traj(self, traj):
        return traj[-1][2].to(self.device)

    @torch.no_grad()
    def eval_step(self, num_samples, logreward_fn=None):
        if logreward_fn is None:
            logreward_fn = self.logr_fn
        self.eval()

        if self.cfg.task in ["cox"]: # save cuda memory
            bs_ls = num_to_groups(num_samples, 250)
            pis_logw = None
            x_max = 0.
            for bs in bs_ls:
                traj_curr, sample_info = sample_traj(self, self.cfg, logreward_fn, batch_size=bs)
                logw_curr = sample_info['pis_logw']
                x_max = max(x_max, sample_info['x_max'])
                if pis_logw is None:
                    pis_logw = logw_curr
                    traj = traj_curr
                else:
                    pis_logw = torch.cat([pis_logw, logw_curr], dim=0)

            print(f"logw_pis={pis_logw.mean().item():.8e}")
            logw = pis_logw
        else:
            traj, sample_info = sample_traj(self, self.cfg, logreward_fn, batch_size=num_samples)
            pis_logw = sample_info['pis_logw']
            x_max = sample_info['x_max']
            logw = self.log_weight(traj)
            print(f"logw={logw.mean().item():.8e}, logw_pis={pis_logw.mean().item():.8e}")

        # pis_logZ = torch.logsumexp(-pis_logw, dim=0) - np.log(num_samples)
        # Z = \int R(x) dx = E_{PF(tau)}[R(x)PB(tau|x)/PF(tau)]]
        logZ_eval = torch.logsumexp(-logw, dim=0) - np.log(num_samples) # (bs,) -> ()
        logZ_elbo = torch.mean(-logw, dim=0)
        info = {"logz": logZ_eval.item(),
                "logz_elbo": logZ_elbo.item(),
                "x_max": x_max}

        return traj, info

    def visualize(self, traj, logreward_fn=None, step=-1):
        state = traj[-1][1].detach().cpu()
        step_str = (f"-{step}" if step >= 0 else "")

        data_ndim = self.data_ndim
        lim = 7
        if self.cfg.task in ["funnel"]:
            data_ndim = 2
            state = state[:, :2]

        if data_ndim == 2:
            dist_img_path = f"dist{step_str}.png"
            viz_sample2d(state, None, dist_img_path, lim=lim)
            viz_sample2d(state, None, f"dist{step_str}.pdf", lim=lim)

            kde_img_path = f"kde{step_str}.png"
            viz_kde2d(state, None, kde_img_path, lim=lim)
            viz_kde2d(state, None, f"kde{step_str}.pdf", lim=lim)

            alpha = 0.8
            n_contour_levels = 20
            def logp_func(x): return -self.task.energy(x.cuda()).cpu()
            contour_img_path = f"contour{step_str}.png"
            viz_contour_sample2d(state, contour_img_path, logp_func, lim=lim, alpha=alpha,
                                 n_contour_levels=n_contour_levels)
            viz_contour_sample2d(state, f"contour{step_str}.pdf", logp_func, lim=lim, alpha=alpha,
                                 n_contour_levels=n_contour_levels)

            img_dict = {"distribution": dist_img_path,
                        "KDE": kde_img_path,
                        "contour":contour_img_path,
                        }

            return img_dict

        if self.cfg.task in ["manywell"]:
            lim = 3
            x13 = state[:, 0:3:2] # 1st and 3rd dimension
            dist_img_path = f"distx13{step_str}.png"
            viz_sample2d(x13, None, dist_img_path, lim=lim)
            kde_img_path = f"kdex13{step_str}.png"
            viz_kde2d(x13, None, kde_img_path, lim=lim)

            alpha = 0.8
            n_contour_levels = 20
            def logp_func(x_2d):
                x = torch.zeros((x_2d.shape[0], self.data_ndim))
                x[:, 0] = x_2d[:, 0]
                x[:, 2] = x_2d[:, 1]
                return -self.task.energy(x)
            contour_img_path = f"contourx13{step_str}.png"
            viz_contour_sample2d(x13, contour_img_path, logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)
            viz_contour_sample2d(x13, f"contourx13{step_str}.pdf", logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)

            x23 = state[:, 1:3] # 2nd and 3rd dimension
            dist_img_path2 = f"distx23{step_str}.png"
            viz_sample2d(x23, None, dist_img_path2, lim=lim)
            viz_kde2d(x23, None, f"kdex23{step_str}.png", lim=lim)

            def logp_func(x_2d):
                x = torch.zeros((x_2d.shape[0], self.data_ndim))
                x[:, 1] = x_2d[:, 0]
                x[:, 2] = x_2d[:, 1]
                return -self.task.energy(x)
            contour_img_path2 = f"contourx23{step_str}.png"
            viz_contour_sample2d(x23, contour_img_path2, logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)
            viz_contour_sample2d(x23, f"contourx23{step_str}.pdf", logp_func, lim=lim, alpha=alpha, n_contour_levels=n_contour_levels)

            return {"distribution": dist_img_path,
                    "distribution2": dist_img_path2,
                    "KDE": kde_img_path,
                    "contour": contour_img_path,
                    "contour2": contour_img_path2,
                    }


class DetailedBalance(GFlowNet):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)
        self.flow = FourierMLP(
            self.data_ndim, 1,
           num_layers=cfg.f_func.num_layers, channels=cfg.f_func.channels,
           zero_init=True
        )
        self.param_ls = [
            {"params": self.f_func.parameters(), "lr": self.cfg.lr},
            {"params": self.flow.parameters(), "lr": self.cfg.zlr},
        ]
        if hasattr(self, "lgv_coef"):
            self.param_ls.append({"params": self.lgv_coef.parameters(), "lr": self.cfg.lr})
        self.optimizer = self.get_optimizer()

    def save(self, path="alg.pt"):
        self.eval()
        save_dict = {
            "f_func": self.f_func.state_dict(),
            "flow": self.flow.state_dict(),
        }
        torch.save(save_dict, path)

    def load(self, path="alg.pt"):
        save_dict = torch.load(path)
        self.f_func.load_state_dict(save_dict["f_func"])
        self.flow.load_state_dict(save_dict["flow"])



def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


class SubTrajectoryBalanceTransitionBased(DetailedBalance):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)

        self.Lambda = float(cfg.subtb_lambda)
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N)) # (N+1, N+1)
        self.register_buffer('coef', coef, persistent=False)

    def train_step(self, traj):
        self.train()
        batch_size = traj[0][1].shape[0]

        xs = [x.to(self.device) for (t, x, r) in traj]
        states = torch.cat(xs, dim=0)  # ((N+1)*b, d)
        states = rearrange(states, "(T1 b) d -> b T1 d", b=batch_size)  # (b, N+1, d)

        ts = [t[None].to(self.device) for (t, x, r) in traj]
        times = torch.cat(ts, dim=0)  # (N+1, 1)
        time_b = repeat(times, "T1 -> T1 one", one=1)  # (N+1, 1)
        # time_coef = (time_b / self.t_end).squeeze(-1)  # (N+1, 1) -> (N+1,)

        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)  # ((N+1)*b, 1)
        logrs = rearrange(logrs, "(T1 b) -> b T1", b=batch_size)  # (b, N+1)

        info = {"logR": logrs[:, -1].mean().item()}
        for b_idx in range(batch_size):
            state_b = states[b_idx]  # (N+1, d)
            log_pf = self.log_pf(time_b[:-1], state_b[:-1], state_b[1:])  # (N,)
            log_pb = self.log_pb(time_b[1:], state_b[1:], state_b[:-1])  # (N,)

            flow_b = self.flow(time_b, state_b).squeeze(-1)  # (N+1, 1) -> (N+1,)

            flow_b = flow_b + logrs[b_idx]  # (N+1,)
            flow_b[-1] = logrs[b_idx][-1]  # (1,)

            diff_logp = log_pf - log_pb # (N, )
            diff_logp_padded = torch.cat(
                (torch.zeros(1).to(diff_logp), diff_logp.cumsum(dim=-1))
            , dim=0) # (N+1,)
            # this means A1[i, j] = diff_logp[i:j].sum(dim=-1)
            A1 = diff_logp_padded.unsqueeze(0) - diff_logp_padded.unsqueeze(1)  # (N+1, N+1)

            A2 = flow_b[:, None] - flow_b[None, :] + A1  # (N+1, N+1)
            if not torch.all(torch.isfinite(A2)):
                import ipdb;
                ipdb.set_trace()

            A2 = (A2 / self.data_ndim).pow(2)  # (N+1, N+1)
            # torch.triu() is useless here
            loss = torch.triu(A2 * self.coef, diagonal=1).sum()
            info["loss_dlogp"] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        info["loss_train"] = loss.item()
        return info


class SubTrajectoryBalanceTrajectoryBased(DetailedBalance):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)

        self.Lambda = float(cfg.subtb_lambda)
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N)) # (T+1, T+1)
        self.register_buffer('coef', coef, persistent=False)

    def get_flow_logp_from_traj(self, traj, debug=False):
        batch_size = traj[0][1].shape[0]
        xs = [x.to(self.device) for (t, x, r) in traj]
        ts = [t[None].to(self.device).repeat(batch_size, 1) for (t, x, r) in traj]  # slightly faster

        state = torch.cat(xs[:-1], dim=0)  # (T*b, d)
        next_state = torch.cat(xs[1:], dim=0)  # (T*b, d)
        time = torch.cat(ts[:-1], dim=0)  # (T*b, 1)
        next_time = torch.cat(ts[1:], dim=0)  # (T*b, 1)
        log_pf = self.log_pf(time, state, next_state)
        log_pb = self.log_pb(next_time, next_state, state)
        log_pf = rearrange(log_pf, "(T b) -> b T", b=batch_size)
        log_pb = rearrange(log_pb, "(T b) -> b T", b=batch_size)

        states = torch.cat(xs, dim=0)  # ((T+1)*b, d)
        times = torch.cat(ts, dim=0)  # ((T+1)*b, 1)
        flows = self.flow(times, states).squeeze(-1)  # ((T+1)*b, 1) -> ((T+1)*b,)
        flows = rearrange(flows, "(T1 b) -> b T1", b=batch_size)  # (b, T+1)

        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)  # ((T+1)*b, 1)
        logrs = rearrange(logrs, "(T1 b) -> b T1", b=batch_size)  # (b, T+1)
        flows = flows + logrs  # (b, T+1)

        logr_terminal = self.logr_from_traj(traj)  # (b,)
        flows[:, -1] = logr_terminal

        logits_dict = {"log_pf": log_pf, "log_pb": log_pb, "flows": flows}
        return logits_dict

    def train_loss(self, traj):
        batch_size = traj[0][1].shape[0]
        logits_dict = self.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = logits_dict["flows"], logits_dict["log_pf"], logits_dict["log_pb"]
        diff_logp = log_pf - log_pb  # (b, T)

        diff_logp_padded = torch.cat(
            (torch.zeros(batch_size, 1).to(diff_logp), diff_logp.cumsum(dim=-1))
            , dim=1)
        # this means A1[:, i, j] = diff_logp[:, i:j].sum(dim=-1)
        A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)  # (b, T+1, T+1)

        A2 = flows[:, :, None] - flows[:, None, :] + A1  # (b, T+1, T+1)
        A2 = (A2 / self.data_ndim).pow(2).mean(dim=0)  # (T+1, T+1)
        # torch.triu() is useless here
        loss = torch.triu(A2 * self.coef, diagonal=1).sum()
        info = {"loss_dlogp": loss.item()}

        logZ_model = self.flow(traj[0][0][None, None].to(self.device),
                               traj[0][1][:1, :].to(self.device)).detach()
        info["logz_model"] = logZ_model.mean().item()
        return loss, info

    def train_step(self, traj):
        self.train()
        loss, info = self.train_loss(traj)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info["loss_train"] = loss.item()
        return info