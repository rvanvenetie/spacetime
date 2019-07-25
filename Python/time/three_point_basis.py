from basis import Basis
from index_set import IndexSet
from indexed_vector import IndexedVector
from interval import Interval

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


def ms2ss(goal_level, labda):
    if labda[0] == 0: return (goal_level, 2**goal_level * labda[1])
    return (goal_level, 2**(goal_level - labda[0]) * (2 * labda[1] + 1))


def ss2ms(labda):
    # Magic
    l, n = labda
    # Special case: we are one of the end points.
    if n in [0, 2**l]:
        return (0, n // (2**l))
    # We are *right* on the middle point.
    elif n == 2**(l - 1):
        return (1, 0)
    # We are to the left of the middle point.
    elif n < 2**(l - 1):
        # Recurse and simply return.
        ll, nn = ss2ms((l - 1, n))
        return (ll + 1, nn)
    # We are to the right of the middle point.
    else:
        # Recurse and *magic*.
        ll, nn = ss2ms((l - 1, n - 2**(l - 1)))
        return (ll + 1, nn + 2**(ll - 1))


class ThreePointBasis(Basis):
    def __init__(self, indices):
        self.indices = indices
        self.ss_indices = [
            IndexSet({
                ms2ss(level, labda)
                for labda in self.indices.until_level(level)
            }) for level in range(0,
                                  self.indices.maximum_level() + 1)
        ]
        assert len(self.ss_indices) == self.indices.maximum_level() + 1

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        return IndexSet({(0, 0), (0, 1)} | {(l, n)
                                            for l in range(1, max_level + 1)
                                            for n in range(2**(l - 1))})

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        return IndexSet({(0, 0), (0, 1)} | {(l, 0)
                                            for l in range(max_level + 1)})

    @staticmethod
    def sort_inorder(multiscale_indices):
        """ Sort a collection of multiscale indices in order of position. """
        return sorted(multiscale_indices, key=lambda labda: position_ms(labda))

    def scaling_support(self, index):
        """ Inefficient but accurate. """
        indices = sorted(self.scaling_indices_on_level(index[0]))
        i = indices.index(index)
        if i == 0:
            return Interval(0, position_ss(indices[i + 1]))
        elif i == len(indices) - 1:
            return Interval(position_ss(indices[i - 1]), 1)

        return Interval(position_ss(indices[i - 1]),
                        position_ss(indices[i + 1]))

    def wavelet_support(self, labda):
        """ Inefficient but accurate. """
        if labda[0] == 0:
            return Interval(0, 1)
        elif labda[0] == 1:
            return Interval(0, 1)

        indices_sorted = ThreePointBasis.sort_inorder(
            self.indices.until_level(labda[0]))
        i = indices_sorted.index(labda)
        if i == 1:
            left_side = position_ms(indices_sorted[i - 1])
        else:
            left_side = position_ms(indices_sorted[i - 2])
        if i == len(indices_sorted) - 2:
            right_side = position_ms(indices_sorted[i + 1])
        else:
            right_side = position_ms(indices_sorted[i + 2])

        return Interval(left_side, right_side)

    def eval_mother_scaling(self, right, x):
        if not right:
            return (1.0 - x) * ((0 <= x) & (x < 1))
        else:
            return x * ((0 <= x) & (x < 1))

    def eval_scaling(self, index, x):
        indices = sorted(self.scaling_indices_on_level(index[0]))
        i = indices.index(index)
        my_pos = position_ss(index)
        support = self.scaling_support(index)
        left_pos, right_pos = support.a, support.b

        res = 0 * x
        if my_pos != right_pos:
            res += 2**(index[0] / 2) * self.eval_mother_scaling(
                False, (x - my_pos) / (right_pos - my_pos))
        if my_pos != left_pos:
            res += 2**(index[0] / 2) * self.eval_mother_scaling(
                True, (x - left_pos) / (my_pos - left_pos))
        return res

    def eval_wavelet(self, labda, x):
        #return sum([
        #    self.eval_scaling(ss_index, x) * qval for (
        #        ss_index,
        #        qval) in zip(self.scaling_siblings(labda), self.Q_block(labda))
        #])
        if labda[0] == 0: return self.eval_scaling(labda, x)

        l, n = labda
        ss_indices = sorted(self.scaling_indices_on_level(l))
        i = ss_indices.index(ms2ss(l, labda))

        result = self.eval_scaling(ss_indices[i], x)
        if n == 0:
            result -= self.eval_scaling(ss_indices[i - 1], x)
        elif (l, n - 1) in self.indices:
            result -= 1 / 2 * self.eval_scaling(ss_indices[i - 1], x)
        else:
            result -= 1 / 3 * self.eval_scaling(ss_indices[i - 1], x)

        if n == 2**(l - 1) - 1:
            result -= self.eval_scaling(ss_indices[i + 1], x)
        elif (l, n + 1) in self.indices:
            result -= 1 / 2 * self.eval_scaling(ss_indices[i + 1], x)
        else:
            result -= 1 / 3 * self.eval_scaling(ss_indices[i + 1], x)
        return result

    def scaling_indices_on_level(self, l):
        if l >= len(self.ss_indices):
            return IndexSet({})
        return self.ss_indices[l]

    def scaling_parents(self, index):
        l, n = index
        assert l > 0
        if n % 2 == 0:
            return [(l - 1, n // 2)]
        else:
            return [(l - 1, n // 2 + i) for i in range(2)]

    def P_block(self, index):
        assert index[0] > 0
        if index[1] % 2 == 0:
            return [1.0 / sq2]
        else:
            return [0.5 / sq2, 0.5 / sq2]

    def scaling_siblings(self, index):
        l, n = index
        # If the singlescale index offset is odd, it must coincide with a
        # multiscale index on this level.
        if n % 2:
            return [(l, n // 2)]
        # If we are the leftmost singlescale index, it can only interact with
        # multiscale index (l, 0).
        if n == 0:
            return [(l, 0)]
        # Same idea for the rightmost singlescale index.
        if n == 2**l:
            return [(l, 2**(l - 1) - 1)]
        # General case: we are between these two multiscale indices.
        return [(l, (n - 1) // 2), (l, n // 2)]

        # Slower code that works for sure...
        return sorted([
            i for i in self.indices.on_level(index[0])
            if index in self.wavelet_siblings(i)
        ])

    def Q_block(self, index):
        l, n = index
        if n % 2:
            return [1.0]
        if n == 0 or n == 2**l:
            return [-1.0]
        # Note the swapping of n+1
        return [
            -1 / 2 if ss2ms((l, n + 1)) in self.indices else -1 / 3,
            -1 / 2 if ss2ms((l, n - 1)) in self.indices else -1 / 3
        ]

        # Slower code that works for sure...
        def mapping(labda):
            if ms2ss(index[0], labda) == index:
                return 1.0
            elif index[1] == 0 or index[1] == 2**index[0]:
                return -1.0
            else:
                return -0.5

        return [mapping(labda) for labda in self.scaling_siblings(index)]

    def scaling_children(self, index):
        return [(index[0] + 1, 2 * index[1] + i) for i in range(-1, 2)]

    def PT_block(self, index):
        return [0.5 / sq2, 1.0 / sq2, 0.5 / sq2]

    def wavelet_siblings(self, index):
        if index[0] == 0: return [(0, 0), (0, 1)]
        return [(index[0], 2 * index[1] + i) for i in range(3)]

    def QT_block(self, index):
        if index == (0, 0):
            return [1.0, 0.0]
        elif index == (0, 1):
            return [0.0, 1.0]

        l, n = index
        if n == 0:  # We are the leftmost wavelet fn
            left = -1.0
        elif (l, n - 1) in self.indices:  # Our left nbr is an index too
            left = -1 / 2
        else:  # Our left nbr is no index.
            left = -1 / 3

        if n == 2**(l - 1) - 1:  # We are the rightmost wavelet fn
            right = -1.0
        elif (l, n + 1) in self.indices:
            right = -1 / 2
        else:
            right = -1 / 3

        return [left, 1.0, right]

    def singlescale_mass(self, l, Pi, Pi_A, d):
        # TODO: Slow but correct..
        indices = sorted(self.scaling_indices_on_level(l))

        def mapping(labda):
            _, n = labda
            i = indices.index(labda)
            res = 0.0
            if n > 0:
                dist_left = indices[i][1] - indices[i - 1][1]
                res += dist_left * (1 / 6 * d[indices[i - 1]] +
                                    1 / 3 * d[indices[i]])
            if n < 2**l:
                dist_right = indices[i + 1][1] - indices[i][1]
                res += dist_right * (1 / 3 * d[indices[i]] +
                                     1 / 6 * d[indices[i + 1]])
            return res

        res = IndexedVector(
            {labda: mapping(labda) if labda in Pi else 0.0
             for labda in Pi_A})
        return res
