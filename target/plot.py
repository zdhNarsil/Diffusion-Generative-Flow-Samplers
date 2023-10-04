import itertools
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch as th
from einops import rearrange

# Inputs are all numpy arrays

########### 1D plot
def traj_plot1d(traj_len, samples, xlabel, ylabel, title="", fsave="img.png"):
    samples = rearrange(samples, "t b d -> b t d").cpu()
    inds = np.linspace(0, samples.shape[1], traj_len, endpoint=False, dtype=int)
    samples = samples[:, inds]
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(np.arange(traj_len), sample.flatten(), marker="x", label=f"sample {i}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fsave)
    plt.close()

def dist_plot1d(samples, nll_target_fn, nll_prior_fn=None,
                fname="img.png", width=8.):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    num_bins = 100
    density, bins = np.histogram(samples, num_bins, density=True)
    ax.plot(bins[1:], density, label="sampled")

    num_bins = 100
    query_x = th.linspace(-width/2, width/2, num_bins).cuda()
    target_unpdf = th.exp(-nll_target_fn(query_x.view(-1, 1)))
    target_norm_pdf = target_unpdf / th.sum(target_unpdf) / width * num_bins # make integral=1.
    np_x = query_x.cpu().numpy()
    ax.plot(np_x, target_norm_pdf.cpu().numpy(), label="target")

    if nll_prior_fn is not None:
        prior_unpdf = th.exp(-nll_prior_fn(query_x.view(-1, 1)))
        prior_norm_pdf = prior_unpdf / th.sum(prior_unpdf) / width * num_bins
        ax.plot(np_x, prior_norm_pdf.cpu().numpy(), label="prior")

    ax.set_xlim(np_x[0], np_x[-1])
    ax.set_ylim(0, 1.5 * th.max(target_norm_pdf).item())
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)
    return fname

@torch.no_grad()
def drift_surface1d(model): # not used
    model.cuda()
    xs = th.linspace(-3.0, 3.0, 120).view(-1, 1).cuda()
    ts = th.linspace(0.0, 0.99, 100).cuda()

    values = []
    for cur_t in ts:
        values.append(model.f_func(cur_t, xs))
    values = th.cat(values, dim=1)

    x, t, zz = map(lambda x: x.cpu().numpy(), [xs, ts, values])
    tt, xx = np.meshgrid(t, x)
    return tt, xx, zz


########### 2D plot
def viz_sample2d(points, title, fsave, lim=7.0, sample_num=50000):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    if title is not None:
        axs.set_title(title)
    axs.plot(
        points[:sample_num, 0],
        points[:sample_num, 1],
        linewidth=0,
        marker=".",
        markersize=1,
    )
    axs.set_xlim(-lim, lim)
    axs.set_ylim(-lim, lim)
    if "pdf" in fsave:
        fig.savefig(fsave, format="pdf", bbox_inches='tight')
    else:
        fig.savefig(fsave, bbox_inches='tight')
    plt.close(fig)
    plt.close()
    plt.cla()
    plt.clf()

def viz_kde2d(points, title, fname, lim=7.0, sample_num=2000):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
    if title is not None:
        ax.set_title(title)
    sns.kdeplot(
        x=points[:sample_num, 0], y=points[:sample_num, 1],
        cmap="coolwarm", fill=True, ax=ax
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # ax.axis("off")
    if "pdf" in fname:
        fig.savefig(fname, format="pdf", bbox_inches='tight')
    else:
        fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def viz_coutour_with_ax(ax, log_prob_func, lim=3.0, n_contour_levels=None):
    grid_width_n_points = 100
    log_prob_min = -1000.0
    # plot_contours
    x_points_dim1 = torch.linspace(-lim, lim, grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def viz_contour_sample2d(points, fname, log_prob_func,
                         lim=3.0, alpha=0.7, n_contour_levels=None):
    # plotting_bounds = (-3, 3) # for manywells
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    viz_coutour_with_ax(ax, log_prob_func, lim=lim, n_contour_levels=n_contour_levels)

    # plot samples
    samples = torch.clamp(points, -lim, lim)
    samples = samples.cpu().detach()
    ax.plot(samples[:, 0], samples[:, 1],
            linewidth=0, marker=".", markersize=1.5, alpha=alpha)

    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)