from . import basis


class DiscConstScaling(basis.Scaling):
    """ Discontinous piecewise constant scaling function.

    The function (l, n) corresponds to the indicator function on element (l,n).
    That is, it has support [2^(-l)*n, 2^(-l)*(n+1)].
    """
    def __init__(self, labda, parents, support):
        assert len(support) == 1
        super().__init__(labda, parents, support)
        self.children = []

        # Register this scaling function in the Element class.
        assert support[0].phi_disc_const is None
        support[0].phi_disc_const = self

    def refine(self):
        if not self.children:
            self.support[0].refine()
            l, n = self.labda
            self.children.append(
                DiscConstScaling((l + 1, 2 * n), self,
                                 [self.support[0].children[0]]))
            self.children.append(
                DiscConstScaling((l + 1, 2 * n + 1), self,
                                 [self.support[0].children[1]]))
        return self.children

    def prolongate(self):
        self.refine()
        return [(self.children[0], 1), (self.children[1], 1)]

    def restrict(self):
        return [(self.parents, 1)]

    def is_full(self):
        return len(self.children) in [0, 2]

    @staticmethod
    def eval_mother(x):
        return (0 <= x) & (x < 1)

    def eval(self, x, deriv=False):
        assert deriv == False
        l, n = self.labda
        return 1.0 * self.eval_mother(2**l * x - n)


class HaarWavelet(basis.Wavelet):
    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if not self.children:
            phi_left, phi_right = [phi for phi, _ in self.single_scale]
            phi_left.refine()
            phi_right.refine()

            l, n = self.labda
            child_left = HaarWavelet((l + 1, 2 * n), self,
                                     [(phi_left.children[0], 1),
                                      (phi_left.children[1], -1)])
            child_right = HaarWavelet((l + 1, 2 * n + 1), self,
                                      [(phi_right.children[0], 1),
                                       (phi_right.children[1], -1)])
            self.children = [child_left, child_right]
        return self.children

    def is_full(self):
        return len(self.children) in [0, 2]


class HaarBasis(basis.Basis):
    # Create static mother scaling function.
    mother_scaling = DiscConstScaling((0, 0), None, [basis.mother_element])
    mother_scalings = [mother_scaling]
    mother_scaling.refine()

    # Create static mother wavelet function.
    mother_wavelet = HaarWavelet((1, 0), None,
                                 [(mother_scaling.children[0], 1),
                                  (mother_scaling.children[1], -1)])
    mother_wavelets = [mother_wavelet]

    def __init__(self):
        pass
