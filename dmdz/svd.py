__all__ = ['SVD']

import warnings
import numpy as np

class SVD(object):

    def __init__(self, svd_rank=0):
        self.X = None
        self.U = None
        self.s = None
        self.V = None
        self.svd_rank = 0

    # TODO purge?
    # @staticmethod
    # def threshold(X, singular_value_threshold=1e-1):
    #     """
    #     Computes the SVD of matrix X and thresholds the rank to get rid of noisy modes.
    #     Default setting is to cutoff any modes whose singular values are beyond 1e-10.
    #
    #     :param X: data matrix
    #     :param singular_value_threshold: float, cutoff value for singular values
    #
    #     :return: X_denoised
    #     """
    #     U, s, Vh = np.linalg.svd(X, full_matrices=False)
    #     cutoff = len(s) - np.searchsorted(s[::-1], singular_value_threshold)
    #     print("Truncating at singular value {}/{} with threshold at {}"
    #           " to reconstruct denoised data matrix X.".format(cutoff, len(s), singular_value_threshold))
    #     X_denoised = U[:, :cutoff] @ np.diag(s[:cutoff]) @ Vh[:cutoff, :]
    #     return X_denoised, cutoff

    @staticmethod
    def cumulative_energy(s, normalize=True):
        cumulative_energy = np.cumsum(s)
        if normalize:
            cumulative_energy = cumulative_energy/s.sum()
        return cumulative_energy

    @staticmethod
    def gavish_donoho_rank(X, s, energy_threshold=0.999999):
        """
        Returns matrix rank for Gavish-Donoho singular value thresholding.
        Reference: https://arxiv.org/pdf/1305.5870.pdf
        """
        beta = X.shape[0]/X.shape[1]
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        cutoff = np.searchsorted(SVD.cumulative_energy(s), energy_threshold)
        rank = np.sum(s > omega*np.median(s[:cutoff]))
        print("Gavish-Donoho rank is {}, computed on {} of {} "
              "singular values such that cumulative energy is {}.".format(rank, cutoff, len(s), energy_threshold))
        return rank

    @staticmethod
    def svd(X, svd_rank=0, full_matrices=False, verbose=False, **kwargs):
        """
        Computes the SVD of matrix X. Defaults to economic SVD.

        :param svd_rank: 0 for Gavish-Donoho threshold, -1 for no truncation, and
            integers [1,infinty) to attempt that truncation.
        :param full_matrices: False is the economic default.

        :return: U, s, V - note that this is V not Vh!

        See documentation for numpy.linalg.svd for more information.
        """
        U, s, V = np.linalg.svd(X, full_matrices=full_matrices, **kwargs)
        V = V.conj().T

        if svd_rank == 0:
            truncation_decision = "Gavish-Donoho"
            rank = SVD.gavish_donoho_rank(X, s)
        elif svd_rank >= 1:
            truncation_decision = "manual"
            if svd_rank < U.shape[1]:
                rank = svd_rank
            else:
                rank = U.shape[1]
                warnings.warn("svd_rank {} exceeds the {} columns of U. "
                              "Using latter value instead".format(svd_rank, U.shape[1]))
        elif svd_rank == -1:
            truncation_decision="no"
            rank = X.shape[1]

        if verbose:
            print("SVD performed with {} truncation, rank {}.".format(truncation_decision, rank))

        return U[:, :rank], s[:rank], V[:, :rank]

    def fit(self, full_matrices=False, **kwargs):
        if self.X is None:
            raise ValueError('SVD instance has no data X for SVD.X')
        else:
            self.U, self.s, self.V = self.svd(self.X, svd_rank=self.svd_rank,
                                              full_matrices=full_matrices, **kwargs)
        print("Computed SVD using svd_rank={}".format(self.svd_rank))