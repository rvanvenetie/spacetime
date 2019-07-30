from basis import Basis

from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from linear_operator import LinearOperator

import numpy as np

sq3 = np.sqrt(3)


class OrthonormalDiscontinuousLinearBasis(Basis):
    """ We have a multiwavelet basis.

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
            l, n = labda
            return {(l - 1, 2 * (n // 4) + i): block[i, n % 4]
                    for i in range(2)}

        def col(labda):
            l, n = labda
            return {(l + 1, 4 * (n // 2) + i): block[n % 2, i]
                    for i in range(4)}

        return LinearOperator(row, col)

    @property
    def Q(self):
        block = np.array([[-1 / 2, -sq3 / 2, 1 / 2, -sq3 / 2], [0, -1, 0, 1]])

        def row(labda):
            l, n = labda
            return {(l, 2 * (n // 4) + i): 2.0**((l - 1) / 2) * block[i, n % 4]
                    for i in range(2)}

        def col(labda):
            l, n = labda
            return {(l, 4 * (n // 2) + i): 2.0**((l - 1) / 2) * block[n % 2, i]
                    for i in range(4)}

        return LinearOperator(row, col)

    def eval_mother_scaling(self, odd, x):
        if odd:
            return sq3 * (2 * x - 1) * ((0 <= x) & (x < 1))
        else:
            return (0 <= x) & (x < 1)

    def eval_scaling(self, labda, x):
        l, n = labda
        return 1.0 * self.eval_mother_scaling(n % 2, 2**l * x - (n // 2))

    def eval_mother_wavelet(self, odd, x):
        if odd:
            return sq3 * (1 - 4 * x) * (
                (0 <= x) & (x < 0.5)) + sq3 * (4 * x - 3) * ((0.5 <= x) &
                                                             (x < 1))
        else:
            return (1 - 6 * x) * ((0 <= x) &
                                  (x < 0.5)) + (5 - 6 * x) * ((0.5 <= x) &
                                                              (x < 1))

    def eval_wavelet(self, labda, x):
        l, n = labda
        if l == 0:
            return 1.0 * self.eval_mother_scaling(n % 2, x)
        else:
            return 2**((l - 1) / 2) * self.eval_mother_wavelet(
                n % 2, 2**(l - 1) * x - (n // 2))

    def wavelet_support(self, labda):
        l, n = labda
        if l == 0:
            assert n in [0, 1]
            return Interval(0, 1)
        else:
            assert 0 <= n < 2 * 2**(l - 1)
            return Interval(2**-(l - 1) * (n // 2), 2**-(l - 1) * (n // 2 + 1))

    def scaling_support(self, labda):
        l, n = labda
        return Interval(2**-l * (n // 2), 2**-l * (n // 2 + 1))

    def scaling_indices_on_level(self, l):
        # TODO: this can be a much smaller set when the grid is non-uniform.
        return SingleLevelIndexSet({(l, n) for n in range(2 * 2**l)})

    @property
    def singlescale_mass(self):
        """ The singlescale orthonormal mass matrix is simply 2**-l * Id. """

        def row(labda):
            l, n = labda
            return {(l, n): 2**-l}

        return LinearOperator(row, None)

    @property
    def singlescale_damping(self):
        """ The singlescale damping matrix int_0^1 phi_i phi_j' dt. """

        def row(labda):
            l, n = labda
            if n % 2 == 0:
                return {(l, n + 1): 2 * sq3}
            else:
                return {}

        return LinearOperator(row, None)
