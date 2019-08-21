from basis import Basis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval
from linear_operator import LinearOperator

from functools import lru_cache
import numpy as np

sq2 = np.sqrt(2)
sq3 = np.sqrt(3)


def position_ms(labda):
    """ Position in [0,1] of a multiscale index labda. """
    if labda[0] == 0:
        assert labda[1] in [0, 1]
        return float(labda[1])
    return float(2**(-labda[0]) + 2**(1 - labda[0]) * labda[1])


def position_ss(labda):
    """ Position in [0,1] of a singlescale index labda. """
    return float(labda[1] / 2.0**labda[0])


def ms2ss(l, labda):
    """ Singlescale index on level l of a multiscale index labda. """
    if labda[0] == 0: return (l, 2**l * labda[1])
    return (l, 2**(l - labda[0]) * (2 * labda[1] + 1))


@lru_cache(maxsize=2**20)
def ss2ms(labda):
    """ Multiscale index of a singlescale index labda.

    This recursive function is slow, could even be linear in time. Also magic.
    We could fix this by tracking the multiscale index of every singlescale
    index. A simpler option could be to cache the results.
    """
    l, n = labda
    # Special case: we are one of the end points.
    if n in [0, 2**l]:
        return (0, n // (2**l))
    # We are *right* on the middle point.
    elif n == 2**(l - 1):
        return (1, 0)

    # Magic from here on.
    # We are to the left of the middle point.
    elif n < 2**(l - 1):
        # Recurse, magic, and simply return.
        ll, nn = ss2ms((l - 1, n))
        return (ll + 1, nn)
    # We are to the right of the middle point.
    else:
        # Recurse and *magic*.
        ll, nn = ss2ms((l - 1, n - 2**(l - 1)))
        return (ll + 1, nn + 2**(ll - 1))


class ThreePointBasis(Basis):
    """ Implements the three-point basis.

    Has a couple of expensive steps but we could fix that to be O(1) time if
    we build something that tracks neighbours of scaling indices.
    """

    def __init__(self, indices, vanish_at_boundary=False):
        super().__init__(indices)

        # TODO: This is a loop with complexity O(max_level * |indices|), oops!
        # Alternative could be to assume that `indices` is in lexicographical
        # ordering, so that we can build this list as we go.
        self.ss_indices = [
            SingleLevelIndexSet({
                ms2ss(level, labda)
                for labda in self.indices.until_level(level)
            }) for level in range(0, self.indices.maximum_level + 1)
        ]
        assert len(self.ss_indices) == self.indices.maximum_level + 1

        # Some memoizations.
        self._scaling_support = {}
        self._wavelet_support = {}

        # Incorporating these ``essential boundary conditions'' is mainly for
        # testing in `ode_test.py` and will not be necessary for our parabolic
        # situation, because the boundary conditions are imposed "weakly" in
        # the sense that it is baked into the variational form rather than the
        # spaces we work with.
        self.vanish_at_boundary = vanish_at_boundary

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0), (0, 1)}
                                  | {(l, n)
                                     for l in range(1, max_level + 1)
                                     for n in range(2**(l - 1))})

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        return MultiscaleIndexSet({(0, 0), (0, 1)}
                                  | {(l, 0)
                                     for l in range(max_level + 1)})

    @property
    def P(self):
        def row(labda):
            l, n = labda
            if n % 2 == 0: return {(l - 1, n // 2): 1.0 / sq2}
            else: return {(l - 1, (n // 2) + i): 0.5 / sq2 for i in range(2)}

        def col(labda):
            l, n = labda
            return {
                (l + 1, 2 * n - 1): 0.5 / sq2,
                (l + 1, 2 * n): 1.0 / sq2,
                (l + 1, 2 * n + 1): 0.5 / sq2
            }

        return LinearOperator(row, col)

    @property
    def Q(self):
        def row(labda):
            l, n = labda
            if (l, n) == (0, 0): return {(0, 0): 1.0}
            if (l, n) == (0, 1): return {(0, 1): 1.0}

            # If the singlescale index offset is odd, it must coincide with a
            # multiscale index on this level.
            if n % 2 == 1: return {(l, n // 2): 1.0}

            # If we are the leftmost singlescale index, it can only interact
            # with multiscale index (l, 0).
            if n == 0:
                return {(l, 0): 0.0 if self.vanish_at_boundary else -1.0}
            # Same idea for the rightmost singlescale index.
            if n == 2**l:
                return {
                    (l, 2**(l - 1) - 1):
                    0.0 if self.vanish_at_boundary else -1.0
                }

            # General case: we are between these two multiscale indices.
            return {
                (l, (n - 1) // 2):
                -1 / 2 if ss2ms((l, n + 1)) in self.indices else -1 / 3,
                (l, n // 2):
                -1 / 2 if ss2ms((l, n - 1)) in self.indices else -1 / 3
            }

        def col(labda):
            l, n = labda
            if (l, n) == (0, 0): return {(0, 0): 1.0}
            if (l, n) == (0, 1): return {(0, 1): 1.0}

            if n == 0: left = 0.0 if self.vanish_at_boundary else -1.0

            elif (l, n - 1) in self.indices: left = -1 / 2
            else: left = -1 / 3

            if n == 2**(l - 1) - 1:
                right = 0.0 if self.vanish_at_boundary else -1.0
            elif (l, n + 1) in self.indices:
                right = -1 / 2
            else:
                right = -1 / 3

            return {
                (l, 2 * n): left,
                (l, 2 * n + 1): 1.0,
                (l, 2 * n + 2): right
            }

        return LinearOperator(row, col)

    def singlescale_mass(self, basis_out=None):
        if basis_out:
            # I have currently only done the case where basis_in == basis_out.
            assert isinstance(basis_out, ThreePointBasis) and self is basis_out

        def row(labda):
            l, n = labda

            left, right = self.scaling_indices_on_level(l).neighbours(labda)

            res = {labda: 0.0}
            if n > 0:
                dist_left = (labda[1] - left[1]) / (2**l)
                res[left] = (dist_left / 6) * 2**l
                res[labda] = res[labda] + (dist_left / 3) * 2**l
            if n < 2**l:
                dist_right = (right[1] - labda[1]) / (2**l)
                res[labda] = res[labda] + (dist_right / 3) * 2**l
                res[right] = (dist_right / 6) * 2**l
            return res

        return LinearOperator(row)

    def singlescale_damping(self, basis_out):
        if isinstance(basis_out, ThreePointBasis):
            # I have only built the case where the in- and out basis are equal.
            assert self is basis_out

            def row(labda):
                l, n = labda
                left, right = self.scaling_indices_on_level(l).neighbours(
                    labda)

                res = {}
                if n == 0: res[labda] = -2**(l - 1)
                else: res[left] = -2**(l - 1)
                if n == 2**l: res[labda] = 2**(l - 1)
                else: res[right] = 2**(l - 1)
                assert None not in res
                return res

            return LinearOperator(row)
        elif isinstance(basis_out, OrthonormalDiscontinuousLinearBasis):
            # TODO: this is probably necessary for the spacetime case.
            return LinearOperator(row=lambda x: {})

    def singlescale_stiffness(self, basis_out=None):
        if basis_out:
            # I have currently only done the case where basis_in == basis_out.
            assert isinstance(basis_out, ThreePointBasis) and self is basis_out

        def row(labda):
            l, n = labda

            left, right = self.scaling_indices_on_level(l).neighbours(labda)

            res = {labda: 0.0}
            if n > 0:
                dist_left = position_ss(labda) - position_ss(left)
                res[left] = -1 / dist_left * 2**l
                res[labda] = res[labda] + 1 / dist_left * 2**l
            if n < 2**l:
                dist_right = position_ss(right) - position_ss(labda)
                res[labda] = res[labda] + 1 / dist_right * 2**l
                res[right] = -1 / dist_right * 2**l
            return res

        return LinearOperator(row)

    def scaling_support(self, labda):
        if labda not in self._scaling_support:
            l, n = labda

            left, right = self.scaling_indices_on_level(l).neighbours(labda)

            if n == 0: return Interval(0, position_ss(right))
            elif n == 2**l: return Interval(position_ss(left), 1)

            self._scaling_support[labda] = Interval(position_ss(left),
                                                    position_ss(right))
        return self._scaling_support[labda]

    def wavelet_support(self, labda):
        if labda not in self._wavelet_support:
            l, n = labda
            if l == 0: return Interval(0, 1)

            left, right = self.scaling_indices_on_level(l).neighbours(
                ms2ss(l, labda))
            if n == 0:
                left_side = 0
            else:
                left_left, _ = self.scaling_indices_on_level(l).neighbours(
                    left)
                left_side = position_ss(left_left)

            if n == 2**(l - 1) - 1:
                right_side = 1
            else:
                _, right_right = self.scaling_indices_on_level(l).neighbours(
                    right)
                right_side = position_ss(right_right)

            self._wavelet_support[labda] = Interval(left_side, right_side)
        return self._wavelet_support[labda]

    def eval_mother_scaling(self, right, x, deriv):
        mask = ((0 <= x) & (x < 1))
        if not deriv:
            if not right: return (1.0 - x) * mask
            else: return x * mask
        else:
            if not right: return -1.0 * mask
            else: return 1.0 * mask

    def eval_scaling(self, labda, x, deriv=False):
        # Slow..
        l, n = labda

        left, right = self.scaling_indices_on_level(l).neighbours(labda)
        my_pos = position_ss(labda)
        support = self.scaling_support(labda)
        left_pos, right_pos = support.a, support.b

        res = 0 * x
        if n > 0:
            chain_rule_constant = 1.0 / (my_pos - left_pos) if deriv else 1.0
            res += chain_rule_constant * 2**(l / 2) * self.eval_mother_scaling(
                True, (x - left_pos) / (my_pos - left_pos), deriv)
        if n < 2**l:
            chain_rule_constant = 1.0 / (right_pos - my_pos) if deriv else 1.0
            res += chain_rule_constant * 2**(l / 2) * self.eval_mother_scaling(
                False, (x - my_pos) / (right_pos - my_pos), deriv)
        return res

    def eval_wavelet(self, labda, x, deriv=False):
        if labda[0] == 0: return self.eval_scaling(labda, x, deriv)

        l, n = labda

        left, right = self.scaling_indices_on_level(l).neighbours(
            ms2ss(l, labda))

        result = self.eval_scaling(ms2ss(l, labda), x, deriv)
        if n == 0:
            if not self.vanish_at_boundary:
                result -= self.eval_scaling(left, x, deriv)
        elif (l, n - 1) in self.indices:
            result -= 1 / 2 * self.eval_scaling(left, x, deriv)
        else:
            result -= 1 / 3 * self.eval_scaling(left, x, deriv)

        if n == 2**(l - 1) - 1:
            if not self.vanish_at_boundary:
                result -= self.eval_scaling(right, x, deriv)
        elif (l, n + 1) in self.indices:
            result -= 1 / 2 * self.eval_scaling(right, x, deriv)
        else:
            result -= 1 / 3 * self.eval_scaling(right, x, deriv)
        return result

    def scaling_indices_on_level(self, l):
        if l >= len(self.ss_indices):
            return SingleLevelIndexSet({})
        return self.ss_indices[l]


def test_3point_ss2ms():
    assert ss2ms((0, 0)) == (0, 0)
    assert ss2ms((0, 1)) == (0, 1)

    assert ss2ms((1, 0)) == (0, 0)
    assert ss2ms((1, 1)) == (1, 0)
    assert ss2ms((1, 2)) == (0, 1)

    assert ss2ms((2, 0)) == (0, 0)
    assert ss2ms((2, 1)) == (2, 0)
    assert ss2ms((2, 2)) == (1, 0)
    assert ss2ms((2, 3)) == (2, 1)
    assert ss2ms((2, 4)) == (0, 1)

    assert ss2ms((3, 0)) == (0, 0)
    assert ss2ms((3, 1)) == (3, 0)
    assert ss2ms((3, 2)) == (2, 0)
    assert ss2ms((3, 3)) == (3, 1)
    assert ss2ms((3, 4)) == (1, 0)
    assert ss2ms((3, 5)) == (3, 2)
    assert ss2ms((3, 6)) == (2, 1)
    assert ss2ms((3, 7)) == (3, 3)
    assert ss2ms((3, 8)) == (0, 1)

    assert ss2ms((4, 0)) == (0, 0)
    assert ss2ms((4, 1)) == (4, 0)
    assert ss2ms((4, 2)) == (3, 0)
    assert ss2ms((4, 3)) == (4, 1)
    assert ss2ms((4, 4)) == (2, 0)
    assert ss2ms((4, 5)) == (4, 2)
    assert ss2ms((4, 6)) == (3, 1)
    assert ss2ms((4, 7)) == (4, 3)
    assert ss2ms((4, 8)) == (1, 0)
    assert ss2ms((4, 9)) == (4, 4)
    assert ss2ms((4, 10)) == (3, 2)
    assert ss2ms((4, 11)) == (4, 5)
    assert ss2ms((4, 12)) == (2, 1)
    assert ss2ms((4, 13)) == (4, 6)
    assert ss2ms((4, 14)) == (3, 3)
    assert ss2ms((4, 15)) == (4, 7)
    assert ss2ms((4, 16)) == (0, 1)

    for l in range(1, 10):
        for n in range(0, 2**l + 1):
            assert ms2ss(l, ss2ms((l, n))) == (l, n)

def test_3point_singlescale_indices():
    multiscale_indices = MultiscaleIndexSet({(0, 0), (0, 1), (1, 0), (2, 0),
                                             (2, 1), (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    assert basis.scaling_indices_on_level(1).indices == {(1, 0), (1, 1),
                                                         (1, 2)}
    assert basis.scaling_indices_on_level(2).indices == {(2, 0), (2, 1),
                                                         (2, 2), (2, 3),
                                                         (2, 4)}
    assert basis.scaling_indices_on_level(3).indices == {
        (3, 0), (3, 1), (3, 2), (3, 4), (3, 6), (3, 7), (3, 8)
    }