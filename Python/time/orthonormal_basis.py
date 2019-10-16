import numpy as np

from . import basis
from ..datastructures.tree import MetaRoot

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
    __slots__ = ['nbr', 'pw_constant']
    order = 1

    def __init__(self, labda, support, parents=None):
        assert len(support) == 1
        super().__init__(labda, support=support, parents=parents)
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
            if not self.pw_constant: return self.nbr.refine()
            self.support[0].refine()
            l, n = self.labda
            P = [self, self.nbr]
            child_elts = self.support[0].children
            self.children = [
                DiscLinearScaling((l + 1, 2 * n + 0), [child_elts[0]], P),
                DiscLinearScaling((l + 1, 2 * n + 1), [child_elts[0]], P),
                DiscLinearScaling((l + 1, 2 * n + 2), [child_elts[1]], P),
                DiscLinearScaling((l + 1, 2 * n + 3), [child_elts[1]], P)
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
        return len(self.children) == 4

    @staticmethod
    def eval_mother(constant, x, deriv):
        if not deriv:
            if constant: return 1.0 * ((0 <= x) & (x < 1))
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
    __slots__ = []
    order = 1

    def __init__(self, labda, single_scale, parents=None):
        super().__init__(labda, single_scale=single_scale, parents=parents)

        for elem in self.support:
            assert elem.psi_ortho[self.index % 2] is None
            elem.psi_ortho[self.index % 2] = self

    def _nbr(self):
        """ Finds the `neighbour` of this wavelet, i.e. its twin brother. """
        _, n = self.labda
        nbr_indices = [1, 0, 3, 2]
        nbr = self.parents[0].children[nbr_indices[n % 4]]
        assert self.support == nbr.support
        assert self.labda != nbr.labda
        return nbr

    def refine(self):
        if not self.children and self.level == 0:
            mother_scalings_children = self.single_scale[0][0].refine()
            # Find all other wavelets on level 0 -- copy of scaling functions.
            parents = self.parents[0].children
            assert len(parents) == 2

            children = [
                OrthonormalWavelet(
                    (1, 0),
                    single_scale=zip(
                        mother_scalings_children,
                        (-1 / 2, -sq3 * 1 / 2, 1 / 2, -sq3 * 1 / 2)),
                    parents=parents),
                OrthonormalWavelet(
                    (1, 1),
                    single_scale=[(mother_scalings_children[1], -1),
                                  (mother_scalings_children[3], 1)],
                    parents=parents)
            ]
            for parent in parents:
                parent.children = children
        elif not self.children and self.level > 0:
            l, n = self.labda
            nbr = self._nbr()
            assert not nbr.children

            # Only invoke on the wavelet of type 0.
            if n % 2: return nbr.refine()

            s = 2**(l / 2)  # scaling
            for i in range(2):
                phi = self.single_scale[2 * i][0]
                # Create child of type 0.
                single_scale = zip(phi.refine(),
                                   (-s / 2, -sq3 * s / 2, s / 2, -sq3 * s / 2))
                self.children.append(
                    OrthonormalWavelet((l + 1, 2 * (n + i)), single_scale,
                                       [self, nbr]))
                # Create child of type 1.
                single_scale = [(phi.children[1], -s), (phi.children[3], s)]
                self.children.append(
                    OrthonormalWavelet((l + 1, 2 * (n + i) + 1), single_scale,
                                       [self, nbr]))
            nbr.children = self.children
        return self.children

    def is_full(self):
        if self.level == 0: return len(self.children) == 2
        else: return len(self.children) == 4


class OrthonormalBasis(basis.Basis):
    # Create mother scaling functions; roots of scaling tree.
    mother_scalings = [
        DiscLinearScaling((0, 0), [basis.mother_element]),
        DiscLinearScaling((0, 1), [basis.mother_element])
    ]
    mother_scalings[0].nbr = mother_scalings[1]
    mother_scalings[1].nbr = mother_scalings[0]

    # Create the root of the wavelet tree -- same as the mother scaling.
    roots_wavelet = [
        OrthonormalWavelet((0, 0), single_scale=[(mother_scalings[0], 1)]),
        OrthonormalWavelet((0, 1), single_scale=[(mother_scalings[1], 1)])
    ]

    # Create the metaroots
    metaroot_wavelet = MetaRoot(roots_wavelet)
    metaroot_scaling = MetaRoot(mother_scalings)

    def __init__(self):
        pass
