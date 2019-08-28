from fractions import Fraction

import numpy as np

from sparse_vector import SparseVector

from .interval import Interval, IntervalSet

sq3 = np.sqrt(3)


def support_to_interval(lst):
    """ Convert a list of tuples (l, n) to an actual interval.

    TODO: Legacy, remove this. """
    lst.sort()
    l, n_a = lst[0]
    _, n_b = lst[-1]
    return Interval(Fraction(n_a, 2**l), Fraction(n_b + 1, 2**l))


#TODO: Misschien moeten we de basis splitten in twee classes for schaling functies en waveletfuncties?
class Basis(object):
    def __init__(self, indices):
        """ Default constructor. """
        self.indices = indices

    @classmethod
    def uniform_basis(cls, max_level):
        """ Constructor for basis with uniform refinement. """
        return cls(indices=cls._uniform_multilevel_indices(max_level))

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        """ MultiscaleIndexSet of indices with uniform refinement. """
        pass

    @classmethod
    def origin_refined_basis(cls, max_level):
        """ Constructor for basis with refinement towards origin. """
        return cls(indices=cls._origin_refined_multilevel_indices(max_level))

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        """ MultiscaleIndexSet of indices with refinement towards origin. """
        pass

    @property
    def P(self):
        """ The matrices {P_l}_l such that Phi_{l-1}^T = Phi_l^T P_l.

        P is a LinearOperator object that implements `matvec()` and `rmatvec()`.
        """
        pass

    @property
    def Q(self):
        """ The matrices {Q_l}_l such that Psi_l^T = Phi_l^T Q_l.
        TODO: Ik denk dat je bedoelt Psi_l = Q_l^T Phi_l.

        Q is a LinearOperator object that implements `matvec()` and `rmatvec()`.
        """
        pass

    def scaling_labda_valid(self, labda):
        """ Returns whether the given labda is valid. """
        pass

    def scaling_support(self, labda):
        """ The support of the scaling function phi_labda.

        The support is given as a list of tuples (l, n), corresponding
        to a (sub)interval [2^-l * n, 2^-l * (n+1)].
        """
        pass

    def scaling_indices_on_level(self, l):
        """ SingleLevelIndexSet of singlescale indices on level l; Delta_l. """
        pass

    def scaling_indices_nonzero_in_nbrhood(self, l, x):
        """ Singlescale indices on level l that are nonzero in [x-eps,x+eps]. """
        assert 0 <= x <= 1
        assert isinstance(x, Fraction) or isinstance(x, int)

    def scaling_mass(self):
        """ Mass matrix applied to scaling functions. """

    def wavelet_labda_valid(self, labda):
        """ Returns whether the given wavelet labda is valid. """
        pass

    def wavelet_support(self, labda):
        """ The support of the wavelet function psi_labda. """
        pass

    def wavelet_indices_on_level(self, l):
        """ SingleLevelIndexSet of multiscale indices on level l; Lambda_l. """
        pass

    def wavelet_nbrhood(self, labda):
        """ The neighbourhood S(psi_labda). Defaults to its support. """
        return self.wavelet_support(labda)

    def eval_scaling(self, labda, x, deriv=False):
        """ Debug method. """
        pass

    def eval_wavelet(self, labda, x, deriv=False):
        """ Debug method.

        TODO: shall we simply use Q_col and eval_scaling for this? """
        pass