__all__ = ['DMD']

import os
import numpy as np
import scipy.linalg

import svgutils.compose as sc
from .svd import SVD
from .plots import plot_eigs, plot_singular_vals, plot_power_spectrum
from .plotsupport import set_figdir
from .timestamp import timestamp

import matplotlib.pyplot as plt
from IPython.display import SVG

class DMD(object):

    def __init__(self, X, tlsq_rank=0, svd_rank=0, exact_dmd=False, optimized_b=False, dt=1):
        # self.snapshots = X  # original snapshot data
        self.dt = dt
        self.timesteps = np.arange(X.shape[1]) * self.dt


        self.svd_X = SVD.svd(X, -1, verbose=False)
        self.X = X
        # self.X, self.threshold_cutoff = SVD.threshold(X) # stores denoised X

        self.svd_X1 = None  # SVD for X1, stored for prediction stage
        self.svd_X2 = None

        self.tlsq_rank = tlsq_rank
        self.svd_rank = svd_rank  # TODO fix naming
        self.svd_actual_rank = None
        self.exact_dmd = exact_dmd
        self.optimized_b = optimized_b
        self.fbDMD = False

        self.eigs = None  # DMD eigenvalues
        self.modes = None  # DMD eigenvectors
        self.Atilde = None  # flow map
        self.amplitudes = None  # DMD mode amplitude vector

    @property
    def amplitudes_mod(self):
        return np.abs(self.amplitudes)

    @property
    def omega(self):
        """
        Returns the continuous-time DMD eigenvalues.
        """
        return np.log(self.eigs)/self.dt


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
    def debias_via_tlsq(X1, X2, tlsq_rank=0):
        """
        Computes the total least square on the augmented [X1 X2]^T matrix.
        :param X1: snapshots 0 through N-2
        :param X2: snapshots 1 through N-1snapshots 0 through N-2
        :param tlsq_rank: rank for total least squares debiasing. Choosing 0 will turn off TLSQ.
        References: https://arxiv.org/pdf/1502.03854.pdf
        """

        if isinstance(tlsq_rank, int) and tlsq_rank > 0:
            Vh = np.linalg.svd(np.append(X1, X2, axis=0), full_matrices=False)[-1]
            tlsq_rank = min(tlsq_rank, Vh.shape[0])
            projector_VVh = Vh[:tlsq_rank, :].conj().T@Vh[:tlsq_rank, :]
            print("TLSQ debiasing: rank {}".format(tlsq_rank))
            return X1@projector_VVh, X2@projector_VVh
        else:
            print("TLSQ debiasing: UNUSED")
            return X1, X2

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

    # noinspection SpellCheckingInspection
    def fit(self, verbose=True):

        print("Computing DMD on X, shape {} by {}.".format(*self.X.shape))

        X1 = self.X[:, :-1]
        X2 = self.X[:, 1:]
        X1, X2 = self.debias_via_tlsq(X1, X2, self.tlsq_rank)

        self.svd_X1 = SVD.svd(X1, self.svd_rank, verbose=verbose)
        U1, s1, V1 = self.svd_X1

        if self.fbDMD:
            print("Running forward/backward DMD.")
            self.svd_X2 = SVD.svd(X2, self.svd_rank, verbose=verbose)
            U2, s2, V2 = self.svd_X2

            if len(s2) != len(s1):
                # TODO this is from pydmd. Enhance it so that it'll run anyway with some fallback.
                raise ValueError(
                    'fbDMD: The optimal truncation produced different number of singular'
                    'values for the X and Y matrix, please specify different svd_rank')

            # fbDMD: create forward/backward low dimension operators fAtilde and bAtilde
            fAtilde = np.linalg.multi_dot([U1.T.conj(), X2, V1])*np.reciprocal(s1)
            bAtilde = np.linalg.multi_dot([U2.T.conj(), X1, V2])*np.reciprocal(s2)
            self.Atilde = scipy.linalg.sqrtm(fAtilde @ np.linalg.inv(bAtilde))
        else:
            # DMD: create forward operator Atilde
            self.Atilde = np.linalg.multi_dot([U1.T.conj(), X2, V1])*np.reciprocal(s1)

        # lowrank eigenvalues and vectors
        self.eigs, eigvecs_Atilde = np.linalg.eig(self.Atilde)
        # retrieve eigenmodes of high dimension operator A
        self.modes = ((X2@V1*np.reciprocal(s1))@eigvecs_Atilde) if self.exact_dmd \
            else (U1@eigvecs_Atilde)

        self.amplitudes = DMD.compute_amplitudes(self.X, self.modes, self.optimized_b)

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
        self.Atilde = self.Atilde[indices, :]  # TODO check this! is it indices,: or :,indices ?
        self.modes = self.modes[:, indices]
        self.amplitudes = self.amplitudes[indices]
        print("Sorted DMD analysis by {}.".format(mode))

    def predict(self, X1):
        """
        Predict X2 given a X1.
        """
        # TODO Test predict functionality
        if self.exact_dmd:
            X2 = np.linalg.multi_dot([self.modes, np.diag(self.eigs), np.linalg.pinv(self.modes), X1])
        else:
            U1 = self.svd_X1[0]
            X2 = np.linalg.multi_dot([U1, self.Atilde, U1.conj().T, X1])
        return X2

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

    # def plot_eigs(self, unit_circle=True, colorbar=True, svd_rank=None, savefig=False, verbose=True):
    #     return plot_eigs(self.eigs, self.amplitudes_mod, unit_circle=unit_circle, svd_rank=svd_rank,
    #                      colorbar=colorbar, savefig=savefig, verbose=verbose)
    #
    # def plot_singular_vals(self, savefig=False, verbose=True):
    #     return plot_singular_vals(self.svd_X[1], savefig=savefig, verbose=verbose)
    #
    # def plot_power_spectrum(self, savefig=False, verbose=True):
    #     return plot_power_spectrum(self.omega, self.amplitudes_mod, savefig=savefig, verbose=verbose)
    #
    # def plot_analysis(self):
    #     figdir = set_figdir(verbose=False)
    #     fig_names = [
    #         self.plot_singular_vals(savefig=True, verbose=False)[-1],
    #         self.plot_eigs(colorbar=False, savefig=True, verbose=False)[-1],
    #         self.plot_power_spectrum(savefig=True, verbose=False)[-1]
    #     ]
    #     svgs = []
    #     for name in fig_names:
    #         print(name)
    #         svgs.append(sc.SVG(name + '.svg', fix_mpl=True))
    #         os.remove("{}.svg".format(name))
    #
    #     sc.Figure(sum(svg.width for svg in svgs), max([svg.height for svg in svgs]),
    #               sc.Panel(
    #                   svgs[0],
    #                   sc.Text("(a)", 6, 16, size=11)
    #               ).move(0, 0),
    #               sc.Panel(
    #                   svgs[1],
    #                   sc.Text("(b)", 6, 16, size=11)
    #               ).move(svgs[0].width, 0),
    #               sc.Panel(
    #                   svgs[2],
    #                   sc.Text("(c)", 6, 16, size=11)
    #               ).move(svgs[0].width + svgs[1].width, 0)
    #               ).save("{}/{}-analysis.svg".format(figdir, timestamp()))
