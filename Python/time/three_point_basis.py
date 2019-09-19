from . import basis


class ContLinearScaling(basis.Scaling):
    """ Continuoues piecewise linear scaling function: `hat functions`.

    The function (l, n) corresponds to the hat function of node 2**-l * n.

    The fields `self.nbr_x` correspond to the neighbour on the left/right.
    The fields `self.child_x` correspond to the children on the left/mid/right.

    The field Element.phi_disc_lin object is ordered by labda, i.e. left, right.
    """
    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)

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
        child = ContLinearScaling((l + 1, 2 * n), [self], child_support)
        self.child_mid = child

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
        child = ContLinearScaling((l + 1, n * 2 - 1), [phi_left, phi_right],
                                  phi_right.support[0].children)
        phi_left.child_right = child
        phi_right.child_left = child

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
    is indexed by 0..2^l-1, corresponding to the odd nodes on level l."""
    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if not self.children:
            phi_left, phi_mid, phi_right = [
                phi for phi, _ in self.single_scale
            ]
            l, n = self.labda
            scaling = 2**((l + 1) / 2)

            # First refine the left part
            phi_children = phi_left.refine_mid(), phi_mid.refine_left(
            ), phi_mid.refine_mid()
            if n == 0:
                child_left = ThreePointWavelet(
                    (l + 1, 2 * n), self,
                    zip(phi_children,
                        (-1 * scaling, 1 * scaling, -1 / 2 * scaling)))
            else:
                child_left = ThreePointWavelet(
                    (l + 1, 2 * n), self,
                    zip(phi_children,
                        (-1 / 2 * scaling, 1 * scaling, -1 / 2 * scaling)))

            # Now refine the right part
            phi_children = phi_mid.refine_mid(), phi_mid.refine_right(
            ), phi_right.refine_mid()
            if n == 2**(l - 1) - 1:
                child_right = ThreePointWavelet(
                    (l + 1, 2 * n + 1), self,
                    zip(phi_children,
                        (-1 / 2 * scaling, 1 * scaling, -1 * scaling)))
            else:
                child_right = ThreePointWavelet(
                    (l + 1, 2 * n + 1), self,
                    zip(phi_children,
                        (-1 / 2 * scaling, 1 * scaling, -1 / 2 * scaling)))

            self.children = [child_left, child_right]
        return self.children

    def is_full(self):
        return len(self.children) in [0, 2]


class ThreePointBasis(basis.Basis):
    # Create static mother scaling functions.
    mother_scalings = [
        ContLinearScaling((0, 0), None, [basis.mother_element]),
        ContLinearScaling((0, 1), None, [basis.mother_element])
    ]
    mother_scalings[0].nbr_right = mother_scalings[1]
    mother_scalings[1].nbr_left = mother_scalings[0]

    # Create the static mother wavelet.
    mother_wavelets = [
        ThreePointWavelet((1, 0), None,
                          [(mother_scalings[0].refine_mid(), -2**(1 / 2)),
                           (mother_scalings[0].refine_right(), 2**(1 / 2)),
                           (mother_scalings[1].refine_mid(), -2**(1 / 2))])
    ]

    def __init__(self):
        pass
