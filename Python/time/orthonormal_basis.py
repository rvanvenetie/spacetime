from basis import Basis

from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from linear_operator import LinearOperator
from math import floor
from fractions import Fraction

import numpy as np

sq3 = np.sqrt(3)


class OrthonormalDiscontinuousLinearBasis(Basis):
    """ Discontinuous piecewise linear basis, with orthonormal wavelets.

    It has two wavelets and scaling functions. Even index-offsets correspond
    with the first, odd with the second.
    """

    def __init__(self, indices):
        super().__init__(indices)

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0), (0, 1)}
                                  | {(l, n)
                                     for l in range(1, max_level + 1)
                                     for n in range(2 * 2**(l - 1))})

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(l, i)
                                   for l in range(max_level + 1)
                                   for i in range(2)})

    @property
    def P(self):
        block = np.array([[1, 0, 1, 0], [-sq3 / 2, 1 / 2, sq3 / 2, 1 / 2]])

        def row(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return {(l - 1, 2 * (n // 4) + i): block[i, n % 4]
                    for i in range(2)}

        def col(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return {(l + 1, 4 * (n // 2) + i): block[n % 2, i]
                    for i in range(4)}

        return LinearOperator(row, col)

    @property
    def Q(self):
        block = np.array([[-1 / 2, -sq3 / 2, 1 / 2, -sq3 / 2], [0, -1, 0, 1]])

        def row(labda):
            assert self.scaling_labda_valid(labda)
            l, n = labda
            return {(l, 2 * (n // 4) + i): 2.0**((l - 1) / 2) * block[i, n % 4]
                    for i in range(2)}

        def col(labda):
            assert self.wavelet_labda_valid(labda)
            l, n = labda
            return {(l, 4 * (n // 2) + i): 2.0**((l - 1) / 2) * block[n % 2, i]
                    for i in range(4)}

        return LinearOperator(row, col)

    def eval_mother_scaling(self, odd, x, deriv):
        if not deriv:
            if odd: return sq3 * (2 * x - 1) * ((0 <= x) & (x < 1))
            else: return (0 <= x) & (x < 1)
        else:
            if odd: return sq3 * 2 * ((0 <= x) & (x < 1))
            else: return 0 * ((0 <= x) & (x < 1))

    def eval_scaling(self, labda, x, deriv=False):
        l, n = labda
        chain_rule_constant = 2**l if deriv else 1.0
        return chain_rule_constant * self.eval_mother_scaling(
            n % 2, 2**l * x - (n // 2), deriv)

    def eval_mother_wavelet(self, odd, x, deriv):
        mask1 = ((0 <= x) & (x < 0.5))
        mask2 = ((0.5 <= x) & (x < 1))
        if not deriv:
            if odd: return sq3 * ((1 - 4 * x) * mask1 + (4 * x - 3) * mask2)
            else: return (1 - 6 * x) * mask1 + (5 - 6 * x) * mask2
        else:
            if odd: return -4 * sq3 * mask1 + 4 * sq3 * mask2
            else: return -6 * mask1 - 6 * mask2

    def eval_wavelet(self, labda, x, deriv=False):
        l, n = labda
        if l == 0:
            return 1.0 * self.eval_mother_scaling(n % 2, x, deriv)
        else:
            chain_rule_constant = 2**(l - 1) if deriv else 1.0
            return chain_rule_constant * 2**(
                (l - 1) / 2) * self.eval_mother_wavelet(
                    n % 2, 2**(l - 1) * x - (n // 2), deriv)

    def wavelet_labda_valid(self, labda):
        l, n = labda
        if l == 0: return self.scaling_labda_valid(labda)
        return l > 0 and  0 <= n < 2 * 2**(l-1)

    def wavelet_support(self, labda):
        assert self.wavelet_labda_valid(labda)
        l, n = labda
        if l == 0: return self.scaling_support(labda)
        else: return [(l, 2 * (n//2)), (l, 2 * (n // 2) + 1)]

    def wavelet_indices_on_level(self, l):
        if l == 0:
            return SingleLevelIndexSet({(0,0), (0,1)})
        else:
            return SingleLevelIndexSet({(l, n) for n in range(2**l)})

    def scaling_labda_valid(self, labda):
        l, n = labda
        return l >= 0 and 0 <= n < 2 * 2 **l

    def scaling_mass(self):
        """ The singlescale orthonormal mass matrix is simply 2**-l * Id. """
        def row(labda):
            l, n = labda
            return {(l, n): 2**-l}
        return LinearOperator(row, None)

    def scaling_damping(self):
        """ The singlescale damping matrix int_0^1 phi_i' phi_j dt. """
        def row(labda):
            l, n = labda
            if n % 2 == 0:
                return {(l, n + 1): 2 * sq3}
            else:
                return {}
        return LinearOperator(row, None)

    def scaling_support(self, labda):
        assert self.scaling_labda_valid(labda)
        l, n = labda
        return [(l, n//2)]

    def scaling_indices_nonzero_in_nbrhood(self, l, x):
        super().scaling_indices_nonzero_in_nbrhood(l, x)

        # treat boundary seperately:
        if x == 0: return SingleLevelIndexSet({(l, 0), (l, 1)})
        elif x == 1: return SingleLevelIndexSet({(l, 2**(l+1)-2), (l, 2**(l+1)-1)})

        # Find the closest node on left of x
        node = floor(x * 2 **l)

        if x * 2 ** l == node:
            # Return two basis functions on left and right of x
            return SingleLevelIndexSet({(l, n) for n in range(2*node-2, 2*node+2)})
        else:
            # Return the two basis functions active on this interval
            return SingleLevelIndexSet({(l, n) for n in range(2*node, 2*node+2)})

    def scaling_indices_on_level(self, l):
        return SingleLevelIndexSet({(l, n) for n in range(2 * 2**l)})
