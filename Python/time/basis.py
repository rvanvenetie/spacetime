from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
import numpy as np

sq3 = np.sqrt(3)


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

    def scaling_indices_on_level(self, l):
        """ SingleLevelIndexSet of singlescale indices on level l; Delta_l. """
        pass

    @property
    def P(self):
        """ The matrices {P_l}_l such that Phi_{l-1}^T = P_l Phi_l^T.
        
        P is a LinearOperator object that implements `matvec()` and `rmatvec()`.
        """
        pass

    @property
    def Q(self):
        """ The matrices {Q_l}_l such that Psi_l^T = Q_l Phi_l^T.

        Q is a LinearOperator object that implements `matvec()` and `rmatvec()`.
        """
        pass

    def scaling_support(self, labda):
        """ The support of the scaling function phi_labda. """
        pass

    def wavelet_support(self, labda):
        """ The support of the wavelet function psi_labda. """
        pass

    def wavelet_nbrhood(self, labda):
        """ The neighbourhood S(psi_labda). Defaults to its support. """
        return self.wavelet_support(labda)

    def eval_scaling(self, labda, x, deriv=False):
        """ Debug method. """
        pass

    def eval_wavelet(self, labda, x, deriv=False):
        """ Debug method. """
        pass
