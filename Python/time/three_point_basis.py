from basis import Basis
from index_set import IndexSet
from interval import Interval


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
    if labda[0] == 0:
        return (goal_level, 2**goal_level * labda[1])
    return (goal_level, 2**(goal_level - labda[0]) * (2 * labda[1] + 1))


def ss2ms(labda):
    l, i = labda
    while i % 2 == 0:
        l -= 1
        i /= 2
    return (l, i)


class ThreePointBasis(Basis):
    def __init__(self, multiscale_indices=None):
        self.indices = multiscale_indices
        self.ss_indices = [
            IndexSet({
                ms2ss(level, labda)
                for labda in self.indices.until_level(level)
            }) for level in range(0,
                                  self.indices.maximum_level() + 1)
        ]

    @staticmethod
    def sort_inorder(multiscale_indices):
        """ Sort a collection of multiscale indices in order of position. """
        return sorted(multiscale_indices, key=lambda labda: position_ms(labda))

    def scaling_support(self, index):
        """ Inefficient but accurate. """
        indices = sorted(self.ss_indices[index[0]])
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
        elif labda[0] == 2:
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
        indices = sorted(self.ss_indices[index[0]])
        i = indices.index(index)
        my_pos = position_ss(index)
        support = self.scaling_support(index)
        left_pos, right_pos = support.a, support.b

        if my_pos == left_pos:
            return 2**(index[0] / 2) * self.eval_mother_scaling(
                False, (x - my_pos) / (right_pos - my_pos))
        elif my_pos == right_pos:
            return 2**(index[0] / 2) * self.eval_mother_scaling(
                True, (x - left_pos) / (my_pos - left_pos))
        return 2**(index[0] /
                   2) * (self.eval_mother_scaling(True, (x - left_pos) /
                                                  (my_pos - left_pos)) +
                         self.eval_mother_scaling(False, (x - my_pos) /
                                                  (right_pos - my_pos)))

    def eval_wavelet(self, labda, x):
        ss_indices = sorted(self.ss_indices[labda[0]])
        i = ss_indices.index(ms2ss(labda[0], labda))

        result = self.eval_scaling(ss_indices[i], x)
        if labda[0] == 0:
            return result

        if i == 1:
            result -= self.eval_scaling(ss_indices[i - 1], x)
        elif ss_indices[i - 2][0] == labda[0]:
            result -= 1 / 2 * self.eval_scaling(ss_indices[i - 1], x)
        else:
            result -= 1 / 3 * self.eval_scaling(ss_indices[i - 1], x)

        if i == len(ss_indices) - 2:
            result -= self.eval_scaling(ss_indices[i + 1], x)
        elif ss_indices[i + 2][0] == labda[0]:
            result -= 1 / 2 * self.eval_scaling(ss_indices[i + 1], x)
        else:
            result -= 1 / 3 * self.eval_scaling(ss_indices[i + 1], x)
        return result

    def scaling_indices_on_level(self, l):
        return self.ss_indices[l]

    def uniform_wavelet_indices(self, max_level):
        return IndexSet({(0, 0), (0, 1)} | {(l, n)
                                            for l in range(1, max_level + 1)
                                            for n in range(2**(l - 1))})

    def origin_refined_wavelet_indices(self, max_level):
        return IndexSet({(0, 0), (0, 1)} | {(l, 0)
                                            for l in range(max_level + 1)})

    def scaling_siblings(self, index):
        # Slow but it works..
        my_support = self.scaling_support(index)
        return sorted([
            i for i in self.indices.on_level(index[0])
            if my_support.intersects(self.wavelet_support(i))
        ])

    def scaling_parents(self, index):
        # Slow but it works..
        my_support = self.scaling_support(index)
        return sorted([
            i for i in self.ss_indices[index[0] - 1]
            if my_support.intersects(self.scaling_support(i))
        ])

    def scaling_children(self, index):
        # Slow but it works..
        my_support = self.scaling_support(index)
        return sorted([
            i for i in self.ss_indices[index[0] + 1]
            if my_support.intersects(self.scaling_support(i))
        ])

    def wavelet_siblings(self, index):
        # Slow but it works..
        my_support = self.wavelet_support(index)
        return sorted([
            i for i in self.ss_indices[index[0]]
            if my_support.intersects(self.scaling_support(i))
        ])
