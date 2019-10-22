from . import basis
from ..datastructures.tree import MetaRoot


class DiscConstScaling(basis.Scaling):
    """ Discontinous piecewise constant scaling function.

    The function (l, n) corresponds to the indicator function on element (l,n).
    That is, it has support [2^(-l)*n, 2^(-l)*(n+1)].
    """
    __slots__ = []
    order = 0

    def __init__(self, labda, support, parents=None):
        assert len(support) == 1
        super().__init__(labda, support=support, parents=parents)

        # Register this scaling function in the Element class.
        assert support[0].phi_disc_const is None
        support[0].phi_disc_const = self

    def refine(self):
        if not self.children:
            self.support[0].refine()
            l, n = self.labda
            self.children.append(
                DiscConstScaling((l + 1, 2 * n),
                                 support=[self.support[0].children[0]],
                                 parents=[self]))
            self.children.append(
                DiscConstScaling((l + 1, 2 * n + 1),
                                 support=[self.support[0].children[1]],
                                 parents=[self]))
        return self.children

    def prolongate(self):
        self.refine()
        return [(self.children[0], 1), (self.children[1], 1)]

    def restrict(self):
        assert len(self.parents) == 1
        return [(self.parents[0], 1)]

    def is_full(self):
        return len(self.children) == 2

    @staticmethod
    def eval_mother(t):
        return (0 <= t) & (t < 1)

    def eval(self, t, deriv=False):
        assert not deriv
        l, n = self.labda
        return 1.0 * self.eval_mother(2**l * t - n)


class HaarWavelet(basis.Wavelet):
    __slots__ = []
    order = 0

    def __init__(self, labda, single_scale, parents=None):
        super().__init__(labda, single_scale=single_scale, parents=parents)

    def refine(self):
        if not self.children and self.level == 0:
            mother_scaling_children = self.single_scale[0][0].refine()
            child = HaarWavelet(
                (1, 0),
                single_scale=[(mother_scaling_children[0], 1),
                              (mother_scaling_children[1], -1)],
                parents=[self])
            self.children = [child]
        elif not self.children and self.level > 0:
            phi_left, phi_right = [phi for phi, _ in self.single_scale]
            phi_left.refine()
            phi_right.refine()

            l, n = self.labda
            child_left = HaarWavelet((l + 1, 2 * n),
                                     single_scale=[(phi_left.children[0], 1),
                                                   (phi_left.children[1], -1)],
                                     parents=[self])
            child_right = HaarWavelet(
                (l + 1, 2 * n + 1),
                single_scale=[(phi_right.children[0], 1),
                              (phi_right.children[1], -1)],
                parents=[self])
            self.children = [child_left, child_right]
        return self.children

    def is_full(self):
        if self.level == 0: return len(self.children) == 1
        else: return len(self.children) == 2


class HaarBasis(basis.Basis):
    # Create mother scaling function; root scaling tree.
    mother_scaling = DiscConstScaling((0, 0), support=[basis.mother_element])

    # Create the root of the wavelet tree -- same as the mother scaling.
    root_wavelet = HaarWavelet((0, 0), single_scale=[(mother_scaling, 1)])

    # Create the metaroots
    metaroot_wavelet = MetaRoot(root_wavelet)
    metaroot_scaling = MetaRoot(mother_scaling)

    def __init__(self):
        pass
