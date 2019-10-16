from . import basis
from ..datastructures.tree import MetaRoot


class ContLinearScaling(basis.Scaling):
    """ Continuoues piecewise linear scaling function: `hat functions`.

    The function (l, n) corresponds to the hat function of node 2**-l * n.

    The fields `self.nbr_x` correspond to the neighbour on the left/right.
    The fields `self.child_x` correspond to the children on the left/mid/right.

    The field Element.phi_cont_lin object is ordered by labda, i.e. left, right.
    """
    __slots__ = [
        'nbr_left', 'nbr_right', 'child_left', 'child_mid', 'child_right'
    ]
    order = 1

    def __init__(self, labda, support, parents=None):
        super().__init__(labda, support=support, parents=parents)

        # TODO: We could read all these properties from the Element directly.
        self.nbr_left = None
        self.nbr_right = None

        self.child_left = None
        self.child_mid = None
        self.child_right = None

        l, n = labda
        if n > 0:
            assert support[0].phi_cont_lin[1] is None
            support[0].phi_cont_lin[1] = self
        if n < 2**l:
            assert support[-1].phi_cont_lin[0] is None
            support[-1].phi_cont_lin[0] = self

    def _update_children(self):
        """ Updates the children variable. Invoke after refining. """
        self.children = list(
            filter(None, [self.child_left, self.child_mid, self.child_right]))

    def refine(self):
        raise TypeError("Regular refinement of ContLinearScaling impossible!")

    def is_full(self):
        raise TypeError("ContLinearScaling functions cannot be `full' or not!")

    def refine_mid(self):
        if self.child_mid: return self.child_mid
        l, n = self.labda

        # Calculate the support of the refined hat.
        for elem in self.support:
            elem.refine()
        child_support = []
        if n > 0: child_support.append(self.support[0].children[1])
        if n < 2**l: child_support.append(self.support[-1].children[0])

        # Create element.
        child = ContLinearScaling((l + 1, 2 * n), child_support, [self])
        self.child_mid = child
        self._update_children()

        # Update nbrs.
        if self.child_left:
            self.child_left.nbr_right = child
            child.nbr_left = self.child_left

        if self.child_right:
            self.child_right.nbr_left = child
            child.nbr_right = self.child_right

        return child

    def refine_left(self):
        """ This refines the hat function on the left. """
        assert self.nbr_left
        if self.child_left:
            assert self.nbr_left.child_right == self.child_left
            return self.child_left

        # Some shortcuts for ease of reading.
        l, n = self.labda
        phi_left = self.nbr_left
        phi_right = self

        # Create child element.
        child = ContLinearScaling(
            (l + 1, n * 2 - 1),
            support=phi_right.support[0].children,
            parents=[phi_left, phi_right],
        )
        phi_left.child_right = child
        phi_right.child_left = child
        phi_left._update_children()
        phi_right._update_children()

        # Update nbrs.
        if phi_left.child_mid:
            phi_left.child_mid.nbr_right = child
            child.nbr_left = phi_left.child_mid

        if phi_right.child_mid:
            phi_right.child_mid.nbr_left = child
            child.nbr_right = phi_right.child_mid

        return child

    def refine_right(self):
        if self.child_right: return self.child_right
        return self.nbr_right.refine_left()

    def prolongate(self):
        l, n = self.labda
        result = [(self.refine_mid(), 1)]
        if n > 0: result.append((self.refine_left(), 0.5))
        if n < 2**l: result.append((self.refine_right(), 0.5))
        return result

    def restrict(self):
        if len(self.parents) == 1:
            return [(self.parents[0], 1)]
        else:
            return [(parent, 0.5) for parent in self.parents]

    @staticmethod
    def eval_mother(x, deriv=False):
        """ Evaluates the hat function on [-1,1] centered at 0. """
        left_mask = (-1 < x) & (x <= 0)
        right_mask = (0 < x) & (x < 1)

        if not deriv:
            return left_mask * (1 + x) + right_mask * (1 - x)
        else:
            return left_mask * 1 + right_mask * -1

    def eval(self, x, deriv=False):
        l, n = self.labda
        c = 2**l if deriv else 1.0
        return c * self.eval_mother(2**l * x - n, deriv)


class ThreePointWavelet(basis.Wavelet):
    """ Continuous piecewise linear basis, with three point wavelets.

    Scaling functions are simply the hat functions. A hat function on given
    level is indexed by the corresponding node index. A wavelet on level l >= 1
    is indexed by 0..2^l-1, corresponding to the odd nodes on level l.
    """
    __slots__ = []
    order = 1

    def __init__(self, labda, single_scale, parents=None):
        super().__init__(labda, single_scale=single_scale, parents=parents)

    def refine(self):
        if not self.children and self.level == 0:
            # Find all other wavelets on level 0 -- copy of scaling functions.
            parents = self.parents[0].children
            assert len(parents) == 2
            assert parents[0].labda == (0, 0)
            assert parents[1].labda == (0, 1)

            # Find the associated single_scale functions.
            mother_scalings = [
                parents[0].single_scale[0][0], parents[1].single_scale[0][0]
            ]

            # Create the mother wavelet.
            sq2 = 2**(1 / 2)
            child = ThreePointWavelet(
                (1, 0),
                single_scale=[(mother_scalings[0].refine_mid(), -sq2),
                              (mother_scalings[0].refine_right(), sq2),
                              (mother_scalings[1].refine_mid(), -sq2)],
                parents=parents)
            for parent in parents:
                parent.children = [child]
        elif not self.children and self.level > 0:
            phi_left, phi_mid, phi_right = [
                phi for phi, _ in self.single_scale
            ]
            l, n = self.labda
            scaling = 2**((l + 1) / 2)

            # First refine the left part
            phi_children = phi_left.refine_mid(), phi_mid.refine_left(
            ), phi_mid.refine_mid()
            if n == 0:
                single_scale = zip(
                    phi_children,
                    (-1 * scaling, 1 * scaling, -1 / 2 * scaling))
            else:
                single_scale = zip(
                    phi_children,
                    (-1 / 2 * scaling, 1 * scaling, -1 / 2 * scaling))

            child_left = ThreePointWavelet((l + 1, 2 * n), single_scale,
                                           [self])

            # Now refine the right part
            phi_children = phi_mid.refine_mid(), phi_mid.refine_right(
            ), phi_right.refine_mid()
            if n == 2**(l - 1) - 1:
                single_scale = zip(
                    phi_children,
                    (-1 / 2 * scaling, 1 * scaling, -1 * scaling))
            else:
                single_scale = zip(
                    phi_children,
                    (-1 / 2 * scaling, 1 * scaling, -1 / 2 * scaling))

            child_right = ThreePointWavelet((l + 1, 2 * n + 1), single_scale,
                                            [self])
            self.children = [child_left, child_right]
        return self.children

    def is_full(self):
        if self.level == 0: return len(self.children) == 1
        else: return len(self.children) == 2


class ThreePointBasis(basis.Basis):
    # Create mother scaling functions; roots of scaling tree.
    mother_scalings = [
        ContLinearScaling((0, 0), [basis.mother_element]),
        ContLinearScaling((0, 1), [basis.mother_element])
    ]
    mother_scalings[0].nbr_right = mother_scalings[1]
    mother_scalings[1].nbr_left = mother_scalings[0]

    # Create the root of the wavelet tree -- same as the mother scaling.
    roots_wavelet = [
        ThreePointWavelet((0, 0), single_scale=[(mother_scalings[0], 1)]),
        ThreePointWavelet((0, 1), single_scale=[(mother_scalings[1], 1)])
    ]

    # Create the metaroots
    metaroot_wavelet = MetaRoot(roots_wavelet)
    metaroot_scaling = MetaRoot(mother_scalings)

    def __init__(self):
        pass
