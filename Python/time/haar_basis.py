from basis import Basis
from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from linear_operator import LinearOperator

import numpy as np


class HaarBasis(Basis):
    def __init__(self, indices):
        super().__init__(indices)
        self.ss_indices = []
        for l in range(0, self.indices.maximum_level + 1):
            Lambda_l = self.indices.on_level(l)
            self.ss_indices.append(
                SingleLevelIndexSet({(l, 2 * n + i)
                                     for (_, n) in Lambda_l
                                     for i in range(2)}))

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
            l, n = labda
            return {(l - 1, n // 2): 1}

        def col(labda):
            l, n = labda
            return {(l + 1, 2 * n): 1, (l + 1, 2 * n + 1): 1}

        return LinearOperator(row, col)

    @property
    def Q(self):
        def row(labda):
            l, n = labda
            return {(l, n // 2): (-1)**n}

        def col(labda):
            l, n = labda
            return {(l, 2 * n): 1, (l, 2 * n + 1): -1}

        return LinearOperator(row, col)

    @property
    def singlescale_mass(self):
        """ The singlescale Haar mass matrix is simply 2**-l * Id. """

        def row(labda):
            l, n = labda
            return {(l, n): 2**-l}

        return LinearOperator(row)

    def eval_mother_scaling(self, x):
        return (0 <= x) & (x < 1)

    def eval_scaling(self, labda, x):
        l, n = labda
        return 1.0 * self.eval_mother_scaling(2**l * x - n)

    def eval_mother_wavelet(self, x):
        return 1.0 * ((0 <= x) & (x < 0.5)) - 1.0 * ((0.5 <= x) & (x < 1.0))

    def eval_wavelet(self, labda, x):
        l, n = labda
        if l == 0:
            return self.eval_mother_scaling(x)
        else:
            return self.eval_mother_wavelet(2**(l - 1) * x - n)

    def wavelet_support(self, labda):
        l, n = labda
        if l == 0:
            assert n == 0
            return Interval(0, 1)
        else:
            assert 0 <= n < 2**(l - 1)
            return Interval(2**-(l - 1) * n, 2**-(l - 1) * (n + 1))

    def scaling_support(self, labda):
        l, n = labda
        return Interval(2**-l * n, 2**-l * (n + 1))

    def scaling_indices_on_level(self, l):
        if l >= len(self.ss_indices):
            return SingleLevelIndexSet({})
        return self.ss_indices[l]
