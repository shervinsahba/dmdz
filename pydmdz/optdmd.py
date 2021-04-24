__all__ = ['OptDMD']

import os
import numpy as np
import scipy.linalg

import svgutils.compose as sc
from .svd import SVD
from .plots import plot_eigs, plot_singular_vals, plot_power_spectrum
from .plotsupport import set_figdir
from .tools import timestamp

from .varpro2 import varpro2, varpro2_opts, varpro2_dexpfun, varpro2_expfun

import matplotlib.pyplot as plt
from IPython.display import SVG


class OptDMD(object):

    def __init__(self, X, timesteps, rank, optimized_b=False):
        self.svd_X = SVD.svd(X, -1, verbose=False)  # TODO check
        self.X = X
        self.timesteps = timesteps
        self.rank = rank  # rank of varpro2 fit, i.e. number of exponentials

        self.optimized_b = optimized_b

        self.eigs = None        # DMD eigenvalues
        self.modes = None       # DMD eigenvectors
        self.amplitudes = None  # DMD mode amplitude vector

    @property
    def amplitudes_mod(self):
        return np.abs(self.amplitudes)

    @property
    def omega(self):
        """
        Returns the continuous-time DMD eigenvalues.
        """
        return np.log(self.eigs)/1 #TODO this is log(eigs)/dt for DMD. What do I consider for OptDMD?

    @property
    def temporaldynamics(self):
        """
        :return: matrix that contains temporal dynamics of each mode, stored by row
        """
        return np.exp(np.outer(self.omega, self.timesteps - self.timesteps[0])) * self.amplitudes[:, None]

    @property
    def reconstruction(self):
        """
        Reconstruction of data matrix X and the mean square error
        """
        reconstruction = (self.modes @ self.temporaldynamics)
        abs_error = np.abs(self.X - reconstruction)
        print("X_dmd MSE {}".format(np.mean(abs_error**2)))
        return reconstruction, abs_error

    @staticmethod
    def compute_amplitudes(X, modes, optimized_b):
        if optimized_b:
            # Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
            # https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
            # TODO. For now, it will return the non-optimized code.
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        else:
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        return b

    @staticmethod
    def optdmd(X, t, r, projected=True, eigs_guess=None, provided_U=None):
        if projected:
            if provided_U is None:
                U, _, _ = np.linalg.svd(X, full_matrices=False)
                U = U[:, :r]
                print('data projection: projected, U_r\'X')
            else:
                U = provided_U
                print('data projection: provided U')
            varpro_X = (U.conj().T@X).T
        else:
            print('data projection: none, X')
            varpro_X = X.T


        if eigs_guess is None:
            def generate_eigs_guess(U, X, t, r):
                UtX = U.conj().T@X;
                UtX1 = UtX[:, :-1]
                UtX2 = UtX[:, 1:]

                dt = np.ravel(t)[1:] - np.ravel(t)[:-1]
                dX = (UtX2 - UtX1)/dt
                Xin = (UtX2 + UtX1)/2

                U1, s1, Vh1 = np.linalg.svd(Xin, full_matrices=False)
                U1 = U1[:, :r]
                V1 = Vh1.conj().T[:, :r]
                s1 = s1[:r]
                Atilde = U1.conj().T@dX@V1/s1

                eigs_guess = np.linalg.eig(Atilde)[0]
                return eigs_guess

            eigs_guess = generate_eigs_guess(U, X, t, r)
            print("eigs_guess: generated eigs seed for varpro2.")
        else:
            print("eigs_guess: user provided eigs seed for varpro2.")

        modes, eigs, eig_array, error, iteration, convergence_status = varpro2(varpro_X, t, varpro2_expfun,
                                                                               varpro2_dexpfun, eigs_guess)
        modes = modes.T

        # normalize
        b = np.sqrt(np.sum(np.abs(modes)**2, axis=0)).T
        indices_small = np.abs(b) < 10*10e-16*max(b)
        b[indices_small] = 1.0
        modes = modes/b
        modes[:, indices_small] = 0.0
        b[indices_small] = 0.0

        if projected:
            modes = U @ modes

        return eigs, modes, b


    def fit(self, projected=True, eigs_guess=None, verbose=True, provided_U=None):
        print("Computing optDMD on X, shape {} by {}.".format(*self.X.shape))
        self.eigs, self.modes, self.amplitudes = OptDMD.optdmd(self.X, self.timesteps, self.rank,
                                                        projected=projected, eigs_guess=eigs_guess, provided_U=provided_U)
        return self


    def sort_by(self, mode="eigs"):
        """
        Sorts DMD analysis results for eigenvalues, eigenvectors (modes, Phi), and amplitudes_mod (b)
        in order of decreasing magnitude, either by "eigs" or "b".
        """
        if mode == "mod_eigs" or mode == "eigs":
            indices = np.abs(self.eigs).argsort()[::-1]
        elif mode == "amplitudes_mod" or mode == "b" or mode == "amps":
            indices = np.abs(self.amplitudes_mod).argsort()[::-1]
        else:
            mode = "default"
            indices = np.arange(len(self.eigs))
        self.eigs = self.eigs[indices]
        self.modes = self.modes[:, indices]
        self.amplitudes = self.amplitudes[indices]
        print("Sorted DMD analysis by {}.".format(mode))

    def predict(self):
        # TODO
        return None

    def plot_eigs(self, marker_s=None, marker_c=None, unit_circle=True, colorbar=True, svd_rank=None,
                   savefig=False, verbose=True, display=True):
        out = plot_eigs(self.eigs, marker_s=marker_s, marker_c=marker_c, unit_circle=unit_circle, svd_rank=svd_rank,
                  colorbar=colorbar, savefig=savefig, verbose=verbose)
        if not display:
            plt.close()
        return out

    def plot_singular_vals(self, savefig=False, verbose=True, display=True):
        out = plot_singular_vals(self.svd_X[1], savefig=savefig, verbose=verbose)
        if not display:
            plt.close()
        return out

    def plot_power_spectrum(self, savefig=False, verbose=True, display=True):
        out = plot_power_spectrum(self.omega, self.amplitudes_mod, savefig=savefig, verbose=verbose)
        if not display:
            plt.close()
        return out

    def plot_analysis(self):
        figdir = set_figdir(verbose=False)
        fig_names = [
            self.plot_singular_vals(savefig=True, verbose=False, display=False)[-1],
            self.plot_eigs(colorbar=False, savefig=True, verbose=False, display=False)[-1],
            self.plot_power_spectrum(savefig=True, verbose=False, display=False)[-1]
        ]
        svgs = []
        for name in fig_names:
            svgs.append(sc.SVG(name + '.svg', fix_mpl=True))
            os.remove("{}.svg".format(name))

        figure_name = "{}/{}-analysis.svg".format(figdir, timestamp())
        sc.Figure(sum(svg.width for svg in svgs), max([svg.height for svg in svgs]),
                  sc.Panel(
                      svgs[0],
                      sc.Text("(a)", 6, 16, size=11)
                  ).move(0, 0),
                  sc.Panel(
                      svgs[1],
                      sc.Text("(b)", 6, 16, size=11)
                  ).move(svgs[0].width, 0),
                  sc.Panel(
                      svgs[2],
                      sc.Text("(c)", 6, 16, size=11)
                  ).move(svgs[0].width + svgs[1].width, 0)
                  ).save(figure_name)
        return SVG(figure_name)