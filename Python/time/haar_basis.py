from basis import Basis
from math import floor
from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from linear_operator import LinearOperator

import numpy as np


class HaarBasis(Basis):
    """ Piecewise constant basis, with haar wavelets.

    Scaling functions are piecewise constants, indexed by the index of the left
    node. The haar wavelets of level l are indexed by the index of the
    left node on level l - 1.
    """
    def __init__(self, indices):
        super().__init__(indices)

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0)} | {(l, n)
                                              for l in range(1, max_level + 1)
                                              for n in range(2**(l - 1))})

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(l, 0) for l in range(max_level + 1)})

    @property
    def P(self):
        def row(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return [((l - 1, n // 2), 1)]

        def col(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return [((l + 1, 2 * n), 1), ((l + 1, 2 * n + 1), 1)]

        return LinearOperator(row, col)

    @property
    def Q(self):
        def row(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return [((l, n // 2), (-1)**n)]

        def col(labda):
            assert self.wavelet_labda_valid(labda)
            l, n = labda
            return [((l, 2 * n), 1), ((l, 2 * n + 1), -1)]

        return LinearOperator(row, col)

    def eval_mother_scaling(self, x):
        return (0 <= x) & (x < 1)

    def eval_scaling(self, labda, x, deriv=False):
        assert deriv == False
        l, n = labda
        return 1.0 * self.eval_mother_scaling(2**l * x - n)

    def eval_mother_wavelet(self, x):
        return 1.0 * ((0 <= x) & (x < 0.5)) - 1.0 * ((0.5 <= x) & (x < 1.0))

    def eval_wavelet(self, labda, x, deriv=False):
        assert deriv == False
        l, n = labda
        if l == 0:
            return self.eval_mother_scaling(x)
        else:
            return self.eval_mother_wavelet(2**(l - 1) * x - n)

    def wavelet_labda_valid(self, labda):
        l, n = labda
        if l == 0: return self.scaling_labda_valid(labda)
        else: return l > 0 and 0 <= n < 2**(l-1)

    def wavelet_support(self, labda):
        assert self.wavelet_labda_valid(labda)
        l, n = labda
        if l == 0: return self.scaling_support(labda)
        return [(l, 2*n), (l,2 * n + 1)]

    def wavelet_indices_on_level(self, l):
        if l == 0:
            return SingleLevelIndexSet({(0,0)})
        else:
            return SingleLevelIndexSet({(l, n) for n in range(2**(l-1))})

    def scaling_labda_valid(self, labda):
        l, n = labda
        return l >= 0 and 0 <= n < 2 **l

    def scaling_mass(self):
        """ The singlescale Haar mass matrix is simply 2**-l * Id. """
        def row(labda):
            l, n = labda
            return [((l, n), 2**-l)]
        return LinearOperator(row)

    def scaling_support(self, labda):
        assert self.scaling_labda_valid(labda)
        return [labda]

    def scaling_indices_nonzero_in_nbrhood(self, l, x):
        super().scaling_indices_nonzero_in_nbrhood(l, x)

        # treat boundary seperately:
        if x == 0: return SingleLevelIndexSet({(l, 0)})
        elif x == 1: return SingleLevelIndexSet({(l, 2**l-1)})

        # Find the closest node on left of x
        node = floor(x * 2 **l)

        # If this node coincides, we have to return the basis on left and right.
        if x * 2 ** l == node: return SingleLevelIndexSet({(l, node-1), (l, node)})
        else: return SingleLevelIndexSet({(l, node)})

    def scaling_indices_on_level(self, l):
        return SingleLevelIndexSet({(l, n) for n in range(2**l)})
