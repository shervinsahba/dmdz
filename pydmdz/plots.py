import matplotlib.pyplot as plt
import numpy as np
from .plotsupport import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

def plot_singular_vals(singular_vals, savefig=False, verbose=True):
    fig, ax = plt.subplots(2,1, sharex=True)

    # semilogy singular values
    ax[0].semilogy(np.arange(len(singular_vals)), singular_vals/np.sum(singular_vals),
                '-o', markeredgecolor='darkslateblue', markerfacecolor='slateblue', markersize=3, markeredgewidth=0.5, linewidth=1.2, alpha=0.3)
    ax[0].grid(alpha=0.2, linestyle=':')
    ax[0].set_ylabel(r'$\sigma_k$', labelpad=-2)
    ax[0].set_ylim(min(singular_vals), max(singular_vals))
    # plt.setp(ax[0].get_xticklabels(), visible=False)
    set_aspect_equal_ratio(ax[0])

    # plot cumulative energy
    cumulative_energy = np.cumsum(singular_vals) / singular_vals.sum()
    ax[1].plot(np.arange(len(cumulative_energy)), cumulative_energy,
                ':o', markeredgecolor='darkslateblue', markerfacecolor='slateblue', markersize=3, markeredgewidth=0.5, linewidth=1.2, alpha=0.3)
    ax[1].grid(alpha=0.2, linestyle=':')
    ax[1].set_ylabel(r'$\sum_{m=1}^k \frac{\sigma_m}{\sum_{n=1}^N\sigma_n}$')
    ax[1].set_xlabel(r'$k$')
    ax[1].set_ylim(0, 1)
    set_aspect_equal_ratio(ax[1])

    # highlight points at various energy levels
    for energy in [0.999999]:
        idx_energy = np.searchsorted(cumulative_energy, energy)
        ax[0].plot(idx_energy, singular_vals[idx_energy]/np.sum(singular_vals), 'o', markeredgecolor='gold', markerfacecolor='darkorange', markersize=3.5, markeredgewidth=0.5, linewidth=1.2, alpha=0.9)
        ax[1].plot(idx_energy, cumulative_energy[idx_energy], 'o', markeredgecolor='gold', markerfacecolor='darkorange', markersize=3.5, markeredgewidth=0.5, linewidth=1.2, alpha=0.9)

    if savefig:
        fig_filename = savefigz("svd", verbose=verbose)
        return fig, ax, fig_filename

    return fig, ax


def plot_eigs(eigs, marker_s=None, marker_c=None, cmap=None, unit_circle=True, colorbar=True, savefig=False, svd_rank=None, verbose=True):

    if svd_rank is None:
        svd_rank = len(eigs)

    if cmap is None:
        cmap="viridis"

    fig, ax = plt.subplots()
    plot = ax.scatter(eigs.real, eigs.imag, s=marker_s, c=marker_c, cmap=cmap, alpha=0.5)
                      #s=amplitudes_mod/4, c=-amplitudes_mod,
                      # TODO create a way to set vmin, vmax or purge
                      # TODO create a way to set marker size s
                      #, vmin=0, vmax=1000)

    if unit_circle:
        LIM_ADJUST = 0.01
        ax.add_artist(
            plt.Circle((0., 0.), 1., color='purple', fill=False,
                       linestyle='--', linewidth=1, alpha=0.5)
        )
        ax.set_xlim(-1 - LIM_ADJUST, 1 + LIM_ADJUST)
        ax.set_ylim(-1 - LIM_ADJUST, 1 + LIM_ADJUST)
        ax.set_aspect('equal')
        xtick_locator(0.5, ax)
        ytick_locator(0.5, ax)

    ax.grid(alpha=0.2, linestyle=':')
    plt.xlabel(r'$\Re\{\lambda\}$')
    plt.ylabel(r'$\Im\{\lambda\}$', labelpad=-4)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar( plot, cax=cax)
        cax.set_ylabel(r'$|b|$', rotation=0)
    if svd_rank:
        plt.text(-0.95, 0.9, "$r=%s$" % svd_rank, fontsize=10)
    if savefig:
        fig_filename = savefigz("eigs", verbose=verbose)
        return fig, ax, fig_filename
    return fig, ax


def plot_power_spectrum(freqs, amplitudes_mod, savefig=False, verbose=True):
    fig, ax = plt.subplots()
    indices = freqs.imag>=0
    freqs = freqs.imag[indices]
    amplitudes_mod = amplitudes_mod[indices]
    plotdata = np.array(sorted(zip(freqs, 2*amplitudes_mod/np.sqrt(len(freqs)))))
    ax.plot(plotdata[:, 0], plotdata[:, 1], '-k', markersize=3.5, alpha=0.2, linewidth=1)
    ax.scatter(plotdata[:, 0], plotdata[:, 1], alpha=0.5, c=-plotdata[:, 1].T, cmap="viridis")
               # TODO keep or purge? , vmin=0, vmax=1000)
    ax.set(xlabel=r'$\Im\{\omega\}$ [Hz]')
    plt.ylabel(r'$|b|\frac{2}{\sqrt{r}}$')
    ax.grid(alpha=0.2, linestyle=':')
    ax.set_ylim(bottom=0)
    set_aspect_equal_ratio(ax)
    if savefig:
        fig_filename = savefigz("powerspectrum", verbose=verbose)
        return fig, ax, fig_filename
    return fig, ax


def plot_mode(modes, savefig=False, verbose=True):
    fig, ax = plt.subplots()
    # plotdata = np.
    return fig, ax