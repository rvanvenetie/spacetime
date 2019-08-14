from basis import Basis
from math import floor
from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from linear_operator import LinearOperator

import numpy as np
from fractions import Fraction


class ThreePointBasis(Basis):
    """ Continuous piecewise linear basis, with three point wavelets.

    Scaling functions are simply the hat functions. A hat function on given
    level is indexed by the corresponding node index. A wavelet on level l >= 1
    is indexed by 0..2^l-1, corresponding to the odd nodes on level l."""
    def __init__(self, indices):
        super().__init__(indices)

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0), (0, 1)}
                                  | {(l, 0)
                                     for l in range(max_level + 1)})

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0), (0, 1)}
                                  | {(l, n)
                                     for l in range(1, max_level + 1)
                                     for n in range(2**(l - 1))})

    @property
    def P(self):
        def row(labda):
            l, n = labda
            if n % 2 == 0:
                return {(l - 1, n // 2): 1.0}
            else:
                return {(l - 1, (n // 2)): 0.5,
                        (l - 1, (n // 2)+1): 0.5}

        def col(labda):
            l, n = labda
            return {
                (l + 1, 2 * n - 1): 0.5,
                (l + 1, 2 * n): 1.0,
                (l + 1, 2 * n + 1): 0.5
            }

        return LinearOperator(row, col)

    @property
    def Q(self):
        def row(labda):
            """ Retrieves a ss labda returns lin. comb. wavelet labdas. """
            l, n = labda

            if l == 0:
                assert 0 <= n <=  1
                return {labda : 1.0}

            assert 0 <= n <= 2**l
            scaling = 2**(l/2)

            # If the singlescale index offset is odd, it must coincide with a
            # multiscale index on this level.
            if n % 2 == 1:
                return {(l, n // 2): 1 * scaling}

            # If we are the leftmost singlescale index, it can only interact
            # with multiscale index (l, 0).
            if n == 0:
                return {(l, 0): -1 * scaling }
            # Same idea for the rightmost singlescale index.
            if n == 2**l:
                return {(l, 2**(l-1)-1): -1 * scaling }

            # General case: we are between these two multiscale indices.
            return {
                (l, (n - 1) // 2): -1/2 * scaling,
                (l, n // 2): -1/2 * scaling
            }

        def col(labda):
            """ Retrieves a wavelet labda returns lin. comb. ss labdas. """
            l, n = labda

            # TODO: this returns a different basis from the one used in followup.pdf
            if l == 0:
                assert 0 <= n <=  1
                return {labda : 1.0}

            # retrieve the node on level l associated to labda
            result = {}
            assert 0 <= n < 2**(l - 1)
            node = 1 + n * 2

            # the wavelets are `scaled` linear combination of normal hat functions.
            scaling = 2**(l/2)
            result[(l, node)] = 1 * scaling

            if node > 1:
                result[(l, node-1)] = -1/2 * scaling
            else:
                result[(l, node-1)] = -1 * scaling

            if node < 2**l - 1:
                result[(l, node+1)] = -1/2 * scaling
            else:
                result[(l, node+1)] = -1 * scaling

            return result

        return LinearOperator(row, col)

    def scaling_mass(self):
        def row(labda):
            l, n = labda

            res = {labda: 0.0}
            if n > 0:
                res[(l,n-1)] = 1/6 * 2**-l
                res[(l,n)] = res[(l,n)] + 1/3 * 2**-l
            if n < 2**l:
                res[(l,n)] = res[(l,n)] + 1/3 * 2**-l
                res[(l,n+1)] = 1/6 * 2**-l
            return res
        return LinearOperator(row)

    def scaling_damping(self):
        def row(labda):
            l, n = labda
            res = {}
            if n == 0:
                res[(l,n)] = -1/2
            else:
                res[(l,n-1)] = -1/2

            if n == 2**l:
                res[(l,n)] = 1/2
            else:
                res[(l,n+1)] = 1/2
            assert None not in res
            return res

        return LinearOperator(row)

    def scaling_stiffness(self):
        def row(labda):
            l, n = labda

            res = {labda: 0.0}
            if n > 0:
                res[(l,n-1)] = -2**l
                res[(l,n)] = res[(l,n)] + 2**l
            if n < 2**l:
                res[(l,n)] = res[(l,n)] + 2**l
                res[(l,n+1)] = -2**l
            return res

        return LinearOperator(row)

    def scaling_support(self, labda):
        l, n = labda
        assert 0 <= n <= 2**l
        h = Fraction(1, 2**l)
        # The boundary scaling functions require special treatment.
        left = max(h * (n-1), 0)
        right = min(h * (n+1), 1)
        return Interval(left, right)

    def scaling_indices_nonzero_in_nbrhood(self, l, x):
        super().scaling_indices_nonzero_in_nbrhood(l, x)
        # treat boundary seperately:
        if x == 0: return SingleLevelIndexSet({(l, 0), (l, 1)})
        elif x == 1: return SingleLevelIndexSet({(l, 2**l-1), (l, 2**l)})

        # Find the closest node on left of x
        node = floor(x * 2 **l)

        if x * 2**l == node:
            return SingleLevelIndexSet({(l, node-1), (l, node), (l, node + 1)})
        else:
            return SingleLevelIndexSet({(l, node), (l, node+1)})

    def scaling_indices_on_level(self, l):
        return SingleLevelIndexSet({(l, n) for n in range(2**l + 1)})

    def wavelet_support(self, labda):
        l, n = labda
        if l == 0:
            assert 0 <= n <=  1
            return Interval(0, 1)
        else:
            assert 0 <= n < 2**(l - 1)
            h = Fraction(1, 2**l)

            # Boundary wavelets require special treatment.
            left = max(h + 2 * h * (n-1), 0)
            right = min(h + 2 * h * (n+1), 1)
            return Interval(left, right)

    def wavelet_indices_on_level(self, l):
        if l == 0:
            return SingleLevelIndexSet({(0, 0), (0, 1)})
        else:
            return SingleLevelIndexSet({(l, n) for n in range(2**(l-1))})


    def eval_mother_scaling(self, x, deriv=False):
        """ Evaluates the hat function on [-1,1] centered at 0. """
        left_mask = (-1 < x) & (x <= 0)
        right_mask = (0 < x) & (x < 1)

        if not deriv:
            return left_mask * (1+x) + right_mask * (1-x)
        else:
            return left_mask * 1 + right_mask * -1

    def eval_scaling(self, labda, x, deriv=False):
        l, n = labda
        c = 2**l if deriv else 1.0
        return c * self.eval_mother_scaling(2**l * x - n, deriv)

    def eval_wavelet(self, labda, x, deriv=False):
        lin_comb_ss = self.Q.col(labda)

        result = 0
        for labda_ss, coeff_ss in lin_comb_ss.items():
            result += coeff_ss * self.eval_scaling(labda_ss, x, deriv)
        return result
