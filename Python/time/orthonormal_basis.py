import numpy as np

from . import basis

sq3 = np.sqrt(3)


class DiscLinearScaling(basis.Scaling):
    """ Discontinous piecewise linear scaling function.

    There are two `types` of scaling functions. Given index (l, n):
    1. For even n, the function represents a piecewise constant on 
       the element (l, n//2).
    2. For odd n, it represents a linear function (from -sqrt(3)
       to sqrt(3)) on the element (l, n // 2).
    
    The field `self.pw_constant` can be used to differentiate between the types.
    The field `self.nbr` can be used to retrieve the `other` scaling function
    on the same element. 
    If set, the field `self.children` contains references to the 4 children,
    in order of labda, i.e. cons left, lin left, cons right, lin right.

    The field Element.phi_disc_lin object is also ordered by labda: cons, lin.
    """
    def __init__(self, labda, parents, support):
        assert len(support) == 1
        super().__init__(labda, parents, support)
        self.children = []
        self.nbr = None  # Store a reference to the `neighbour` on this element.
        self.pw_constant = labda[1] % 2 == 0  # Store type of this function.

        # Register this scaling function in the corresponding elements.
        if self.pw_constant:
            assert support[0].phi_disc_lin[0] is None
            support[0].phi_disc_lin[0] = self
        else:
            assert support[0].phi_disc_lin[1] is None
            support[0].phi_disc_lin[1] = self

    def refine(self):
        if not self.children:
            if not self.pw_constant:
                return self.nbr.refine()
            self.support[0].refine()
            l, n = self.labda
            P = [self, self.nbr]
            child_elts = self.support[0].children
            self.children = [
                DiscLinearScaling((l + 1, 2 * n + 0), P, [child_elts[0]]),
                DiscLinearScaling((l + 1, 2 * n + 1), P, [child_elts[0]]),
                DiscLinearScaling((l + 1, 2 * n + 2), P, [child_elts[1]]),
                DiscLinearScaling((l + 1, 2 * n + 3), P, [child_elts[1]])
            ]
            self.nbr.children = self.children

            # Update neighbouring relations.
            self.children[0].nbr = self.children[1]
            self.children[1].nbr = self.children[0]
            self.children[2].nbr = self.children[3]
            self.children[3].nbr = self.children[2]
        return self.children

    def prolongate(self):
        self.refine()
        if self.pw_constant:
            return [(self.children[0], 1), (self.children[2], 1)]
        else:
            return list(zip(self.children, (-sq3 / 2, 1 / 2, sq3 / 2, 1 / 2)))

    def restrict(self):
        l, n = self.labda
        if n % 4 == 0:
            return [(self.parents[0], 1), (self.parents[1], -sq3 / 2)]
        elif n % 4 == 1:
            return [(self.parents[1], 1 / 2)]
        elif n % 4 == 2:
            return [(self.parents[0], 1), (self.parents[1], sq3 / 2)]
        else:
            return [(self.parents[1], 1 / 2)]

    def is_full(self):
        return len(self.children) in [0, 4]

    @staticmethod
    def eval_mother(constant, x, deriv):
        if not deriv:
            if constant: return (0 <= x) & (x < 1)
            else: return sq3 * (2 * x - 1) * ((0 <= x) & (x < 1))
        else:
            if constant: return 0 * ((0 <= x) & (x < 1))
            else: return sq3 * 2 * ((0 <= x) & (x < 1))

    def eval(self, x, deriv=False):
        l, n = self.labda
        chain_rule_constant = 2**l if deriv else 1.0
        return chain_rule_constant * self.eval_mother(
            self.pw_constant, 2**l * x - (n // 2), deriv)


class OrthonormalWavelet(basis.Wavelet):
    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if not self.children:
            l, n = self.labda
            s = 2**(l / 2)  # scaling
            for i in range(2):
                if n % 2 == 0:
                    phi = self.single_scale[2 * i][0]
                    phi.refine()
                    self.children.append(
                        OrthonormalWavelet(
                            (l + 1, 2 * (n + i)), self,
                            zip(phi.children,
                                (-s / 2, -sq3 * s / 2, s / 2, -sq3 * s / 2))))
                else:
                    phi = self.single_scale[i][0]
                    phi.refine()
                    self.children.append(
                        OrthonormalWavelet((l + 1, 2 * (n + i) - 1), self,
                                           [(phi.children[1], -s),
                                            (phi.children[3], s)]))
        return self.children

    def is_full(self):
        return len(self.children) in [0, 4]


class OrthonormalBasis(basis.Basis):
    # Create static mother scaling function.
    mother_scalings = [
        DiscLinearScaling((0, 0), None, [basis.mother_element]),
        DiscLinearScaling((0, 1), None, [basis.mother_element])
    ]
    mother_scalings[0].nbr = mother_scalings[1]
    mother_scalings[1].nbr = mother_scalings[0]
    mother_scalings[0].refine()
    mother_scalings[1].refine()
    assert (len(mother_scalings[0].children) == 4)
    assert (len(mother_scalings[1].children) == 4)

    # Create static mother wavelet function.
    mother_wavelets = [
        OrthonormalWavelet((1, 0), None,
                           zip(mother_scalings[0].children,
                               (-1 / 2, -sq3 * 1 / 2, 1 / 2, -sq3 * 1 / 2))),
        OrthonormalWavelet((1, 1), None, [(mother_scalings[1].children[1], -1),
                                          (mother_scalings[0].children[3], 1)])
    ]

    def __init__(self):
        pass
