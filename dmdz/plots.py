import matplotlib.pyplot as plt
import numpy as np
from .plotsupport import *
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

LIM_ADJUST = 0.01  # to provide a buffer around plot limits


def plot_singular_vals(singular_vals, savefig=False, verbose=True):
    fig, ax = plt.subplots(2, 1, sharex=True)

    # semilogy singular values
    ax[0].semilogy(np.arange(len(singular_vals)), singular_vals/np.sum(singular_vals),
                   '-o', markeredgecolor='darkslateblue', markerfacecolor='slateblue',
                   markersize=3, markeredgewidth=0.5, linewidth=1.2, alpha=0.3)
    ax[0].tick_params(axis='both', length=2)
    ax[0].grid(alpha=0.2, linestyle=':')
    ax[0].set_ylabel(r'$\sigma_k$', labelpad=-2)
    ax[0].set_ylim(min(singular_vals), max(singular_vals))
    ax[0].set_xlim(left=0)
    # plt.setp(ax[0].get_xticklabels(), visible=False)
    set_aspect_equal_ratio(ax[0])

    # plot cumulative energy
    cumulative_energy = np.cumsum(singular_vals) / singular_vals.sum()
    ax[1].plot(np.arange(len(cumulative_energy)), cumulative_energy,
               ':o', markeredgecolor='darkslateblue', markerfacecolor='slateblue',
               markersize=3, markeredgewidth=0.5, linewidth=1.2, alpha=0.3)
    ax[1].tick_params(axis='both', length=2)
    ax[1].grid(alpha=0.2, linestyle=':')
    ax[1].set_ylabel('energy')
    ax[1].set_xlabel('$k$')
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(left=0)
    set_aspect_equal_ratio(ax[1])

    # highlight points at various energy levels
    for energy in [0.999999]:
        idx_energy = np.searchsorted(cumulative_energy, energy)
        ax[0].plot(idx_energy, singular_vals[idx_energy]/np.sum(singular_vals), 'o',
                   markeredgecolor='gold', markerfacecolor='darkorange', markersize=3.5,
                   markeredgewidth=0.5, linewidth=1.2, alpha=0.9)
        ax[1].plot(idx_energy, cumulative_energy[idx_energy], 'o', markeredgecolor='gold',
                   markerfacecolor='darkorange', markersize=3.5, markeredgewidth=0.5, linewidth=1.2, alpha=0.9)

    if savefig:
        fig_filename = savefigz("svd", verbose=verbose)
        return fig, ax, fig_filename
    return fig, ax


def plot_eigs(eigs, mode, marker_s=None, marker_c=None, cmap=None, colorbar=False,
              svd_rank=None, savefig=False, fit_parabola=True, verbose=True):

    if svd_rank is None:
        svd_rank = len(eigs)

    if cmap is None:
        cmap = sns.color_palette("mako_r", as_cmap=True)

    fig, ax = plt.subplots()

    if marker_s is not None:
        idx = np.argsort(marker_s)
        plot = ax.scatter(eigs.real[idx], eigs.imag[idx], s=marker_s[idx], c=marker_c[idx], cmap=cmap, alpha=0.7)
    else:
        plot = ax.scatter(eigs.real, eigs.imag, s=marker_s, c=marker_c, cmap=cmap, alpha=0.7)

    # TODO create a way to set vmin, vmax for colorbar maybe in plot

    ax.grid(alpha=0.2, linestyle=':')
    ax.tick_params(axis='both', length=2)

    if mode == 'continuous':
        plt.axvline(x=0, color='k', linewidth=1, linestyle='--')
        ax.set_xlim(-1 - LIM_ADJUST, 1 + LIM_ADJUST)
        plt.xlabel(r'$\Re[\omega]$')
        plt.ylabel(r'$\Im[\omega]$', labelpad=-4)
        xtick_locator(1, ax)
        set_aspect_equal_ratio(ax)

        if fit_parabola:
            xp = np.linspace(np.min(eigs.imag), np.max(eigs.imag), 100)
            pfit = np.poly1d(np.polyfit(eigs.imag, eigs.real, 2))
            xp_shift = eigs.real[idx][-1]  # TODO this idx is out of scope. It depends on amplitudes_mod.
            print(f"parabolic fit: {pfit}, recentered to origin and retranslated by {xp_shift}")
            ax.plot(pfit(xp) - pfit(0) + xp_shift, xp, color='b', linewidth=1, linestyle='--')
            plt.text(0.65, 0.85, f"$p={pfit.coefficients[0]:0.3f}\omega_2 {xp_shift:+0.3f}$", fontsize=11, transform=ax.transAxes)


    elif mode == 'discrete':
        ax.add_artist(plt.Circle((0., 0.), 1., color='purple', fill=False,
                                 linestyle='--', linewidth=1, alpha=0.5))
        ax.set_xlim(-1 - LIM_ADJUST, 1 + LIM_ADJUST)
        ax.set_ylim(-1 - LIM_ADJUST, 1 + LIM_ADJUST)
        ax.set_aspect('equal')
        plt.xlabel(r'$\Re[\lambda]$')
        plt.ylabel(r'$\Im[\lambda]$', labelpad=-4)
        xtick_locator(1, ax)
        ytick_locator(1, ax)


    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(plot, cax=cax)
        cax.set_ylabel(r'$|b|$', rotation=0)

    if svd_rank:
        plt.text(0.65, 0.9, f"$r={svd_rank}$", fontsize=11, transform=ax.transAxes)

    if savefig:
        fig_filename = savefigz("eigs", verbose=verbose)
        return fig, ax, fig_filename
    return fig, ax


def plot_power_spectrum(freqs, amplitudes_mod, marker_s=None, marker_c=None, savefig=False, verbose=True, cmap=None):

    if cmap is None:
        cmap = sns.color_palette("mako_r", as_cmap=True)

    fig, ax = plt.subplots()
    indices = freqs.imag >= 0
    freqs = freqs.imag[indices]
    amplitudes_mod = amplitudes_mod[indices]

    if marker_s is None:
        marker_s = 50*np.clip(amplitudes_mod/np.max(amplitudes_mod), 0.2, 1.0)
    if marker_c is None:
        marker_c = amplitudes_mod/np.max(amplitudes_mod)

    plotdata = np.array(sorted(zip(freqs, 2*amplitudes_mod/np.sqrt(len(freqs)))))
    ax.plot(plotdata[:, 0], plotdata[:, 1], '-k', alpha=0.3, linewidth=1)
    ax.scatter(plotdata[:, 0], plotdata[:, 1], s=marker_s, c=marker_c, cmap=cmap, alpha=0.7)
    # TODO vmin=0, vmax=1000 settings?

    ax.set(xlabel=r'$\Im[\omega]$')
    plt.ylabel(r'$|b|\cdot2/\sqrt{r}$')
    ax.tick_params(axis='both', length=2)
    ax.grid(alpha=0.2, linestyle=':')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    set_aspect_equal_ratio(ax)
    if savefig:
        fig_filename = savefigz("powerspectrum", verbose=verbose)
        return fig, ax, fig_filename
    return fig, ax


def plot_mode():
    pass
    return
