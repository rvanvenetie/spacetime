from fractions import Fraction

import numpy as np

from interval import Interval
from linear_operator import LinearOperator

sq3 = np.sqrt(3)


def support_to_interval(lst):
    """ Convert a list of tuples (l, n) to an actual interval.

    TODO: Legacy, remove this. """
    return Interval(lst[0].interval.a, lst[-1].interval.b)


class MultiscaleIndices:
    """ Immutable set of multiscale indices.  """

    def __init__(self, indices):
        self.indices = indices
        self.maximum_level = max([fn.labda[0] for fn in indices])
        self.per_level = [[] for _ in range(self.maximum_level + 1)]
        for fn in indices:
            self.per_level[fn.labda[0]].append(fn)

    def __repr__(self):
        return r"MSIS(%s)" % self.indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def __contains__(self, index):
        return self.indices.__contains__(index)

    def on_level(self, level):
        if level > self.maximum_level: return []
        return self.per_level[level]

    def until_level(self, level):
        """ Expensive (but linear in size) method. Mainly for testing. """
        return MultiscaleIndices(
            {index for index in self.indices if index.labda[0] <= level})


class Element:

    def __init__(self, level, node_index, parent):
        self.level = level
        self.node_index = node_index
        self.parent = parent
        self.children = []

        # Add some extra variables necessary for applicator.
        # This should definitely be somewhere else..
        self.Lambda_in = False
        self.Lambda_out = False
        self.Pi_in = False
        self.Pi_out = False

    def bisect(self):
        if self.children: return
        child_left = self.__class__(self.level + 1, self.node_index * 2, self)
        child_right = self.__class__(self.level + 1, self.node_index * 2 + 1,
                                     self)
        self.children = [child_left, child_right]

    @property
    def interval(self):
        h = Fraction(1, 2**self.level)
        return Interval(h * self.node_index, h * (self.node_index + 1))

    def __repr__(self):
        return 'Element({}, {})'.format(self.level, self.node_index)


# Initializes a mother_element to be used for all bases.
Element.mother_element = Element(0, 0, None)


class BaseScaling:

    def __init__(self, labda, parents, support):
        self.labda = labda
        self.parents = parents
        self.support = support  # Support is a list.
        self.multi_scale = []  # Tranpose of the wavelet to multiscale.
        # TODO: This should be removed, or neatly integrated.
        self.coeff = [0 for _ in range(3)]

    def reset_coeff(self):
        # TODO: This should be removed, or neatly integrated.
        self.coeff = [0 for _ in range(3)]

    def prolongate(self):
        """ Returns a list of pairs with the corresponding coefficients. """
        pass

    def restrict(self):
        """ The adjoint of transposing. """
        pass

    def mass(self):
        """ Mass inproduct for this scaling element. """
        pass

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, *self.labda)


class BaseWavelet:

    def __init__(self, labda, parents, single_scale):
        self.labda = labda
        self.parents = parents
        self.single_scale = list(single_scale)
        self.children = []
        self.support = []
        for phi, coeff in self.single_scale:
            self.support.extend(phi.support)
            # Register this wavelet in the corresponding phi.
            phi.multi_scale.append((self, coeff))

        # TODO: This should be removed, or neatly integrated.
        self.coeff = [0 for _ in range(3)]

    def reset_coeff(self):
        # TODO: This should be removed, or neatly integrated.
        self.coeff = [0 for _ in range(3)]

    def eval(self, x, deriv=False):
        result = 0
        for phi, coeff_ss in self.single_scale:
            result += coeff_ss * phi.eval(x, deriv)
        return result

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, *self.labda)


class BaseBasis:

    @classmethod
    def uniform_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        # Start refining all wavelets
        Lambda = [] + basis.mother_wavelets
        Lambda_l = Lambda
        for _ in range(max_level - 1):
            Lambda_new = []
            for psi in Lambda_l:
                psi.refine()
                Lambda_new.extend(psi.children)
            Lambda_l = Lambda_new
            Lambda.extend(Lambda_l)

        Delta = basis.mother_scalings + list(
            {phi for psi in Lambda for phi, _ in psi.single_scale})
        Lambda = basis.mother_scalings + Lambda
        return basis, MultiscaleIndices(Lambda), MultiscaleIndices(Delta)

    @classmethod
    def origin_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        n_wavelets = len(basis.mother_wavelets)
        Lambda = [] + basis.mother_wavelets
        Lambda_l = Lambda
        for _ in range(max_level - 1):
            parents = list(Lambda[-n_wavelets:])
            for parent in parents:
                parent.refine()
                Lambda.append(parent.children[0])

        Delta = basis.mother_scalings + list(
            {phi for psi in Lambda for phi, _ in psi.single_scale})
        Lambda = basis.mother_scalings + Lambda
        return basis, MultiscaleIndices(Lambda), MultiscaleIndices(Delta)

    @property
    def P(self):

        def row(phi):
            return phi.restrict()

        def col(phi):
            return phi.prolongate()

        return LinearOperator(row, col)

    @property
    def Q(self):

        def row(phi):
            return phi.multi_scale

        def col(psi):
            return psi.single_scale

        return LinearOperator(row, col)

    @staticmethod
    def scaling_mass():

        def row(phi):
            return phi.mass()

        return LinearOperator(row)


class DiscConstScaling(BaseScaling):

    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)
        self.children = []

    def refine(self):
        if self.children: return self.children
        self.support[0].bisect()
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

    def mass(self):
        """ The singlescale Haar mass matrix is simply 2**-l * Id. """
        l, n = self.labda
        return [(self, 2**-l)]

    @staticmethod
    def eval_mother(x):
        return (0 <= x) & (x < 1)

    def eval(self, x, deriv=False):
        assert deriv == False
        l, n = self.labda
        return 1.0 * self.eval_mother(2**l * x - n)


class HaarWavelet(BaseWavelet):

    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if self.children: return
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


class HaarBasis(BaseBasis):
    # Create static mother scaling function.
    mother_scaling = DiscConstScaling((0, 0), None, [Element.mother_element])
    mother_scalings = [mother_scaling]
    mother_scaling.refine()

    # Create static mother wavelet function.
    mother_wavelet = HaarWavelet((1, 0), None,
                                 [(mother_scaling.children[0], 1),
                                  (mother_scaling.children[1], -1)])
    mother_wavelets = [mother_wavelet]

    def __init__(self):
        pass


class DiscLinearScaling(BaseScaling):
    """ Discontinuous scaling functions.

    There are two types, the (l, n) with even n correspond to pw constants,
    whereas the (l, n) with odd n correspond to linears.
    """

    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)
        self.children = []
        self.nbr = None  # Store a reference to the `neighbour` on this element.
        self.pw_constant = labda[1] % 2 == 0  # Store type of this function.

    def refine(self):
        if self.children: return self.children
        if not self.pw_constant: return self.nbr.refine()
        self.support[0].bisect()
        l, n = self.labda
        parents = [self, self.nbr]
        child_left_cons = DiscLinearScaling((l + 1, 2 * n), parents,
                                            [self.support[0].children[0]])
        child_left_lin = DiscLinearScaling((l + 1, 2 * n + 1), parents,
                                           [self.support[0].children[0]])
        child_right_cons = DiscLinearScaling((l + 1, 2 * n + 2), parents,
                                             [self.support[0].children[1]])
        child_right_lin = DiscLinearScaling((l + 1, 2 * n + 3), parents,
                                            [self.support[0].children[1]])

        self.children = [
            child_left_cons, child_left_lin, child_right_cons, child_right_lin
        ]
        self.nbr.children = self.children

        # Update neighbouring relations.
        child_left_cons.nbr = child_left_lin
        child_left_lin.nbr = child_left_cons
        child_right_cons.nbr = child_right_lin
        child_right_lin.nbr = child_right_cons
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

    def mass(self):
        """ The singlescale Haar mass matrix is simply 2**-l * Id. """
        l, n = self.labda
        return [(self, 2**-l)]

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


class OrthoWavelet(BaseWavelet):

    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if self.children: return self.children
        l, n = self.labda
        s = 2**(l / 2)  # scaling
        for i in range(2):
            if n % 2 == 0:
                phi = self.single_scale[2 * i][0]
                phi.refine()
                self.children.append(
                    OrthoWavelet(
                        (l + 1, 2 * (n + i)), self,
                        zip(phi.children,
                            (-s / 2, -sq3 * s / 2, s / 2, -sq3 * s / 2))))
            else:
                phi = self.single_scale[i][0]
                phi.refine()
                self.children.append(
                    OrthoWavelet((l + 1, 2 * (n + i) - 1), self,
                                 [(phi.children[1], -s), (phi.children[3], s)]))


class OrthoBasis(BaseBasis):
    # Create static mother scaling function.
    mother_scalings = [
        DiscLinearScaling((0, 0), None, [Element.mother_element]),
        DiscLinearScaling((0, 1), None, [Element.mother_element])
    ]
    mother_scalings[0].nbr = mother_scalings[1]
    mother_scalings[1].nbr = mother_scalings[0]
    mother_scalings[0].refine()
    mother_scalings[1].refine()
    assert (len(mother_scalings[0].children) == 4)
    assert (len(mother_scalings[1].children) == 4)

    # Create static mother wavelet function.
    mother_wavelet = HaarWavelet((1, 0), None,
                                 [(mother_scalings[0].children[0], 1),
                                  (mother_scalings[0].children[1], -1)])
    mother_wavelets = [
        OrthoWavelet((1, 0), None,
                     zip(mother_scalings[0].children,
                         (-1 / 2, -sq3 * 1 / 2, 1 / 2, -sq3 * 1 / 2))),
        OrthoWavelet((1, 1), None, [(mother_scalings[1].children[1], -1),
                                    (mother_scalings[0].children[3], 1)])
    ]

    def __init__(self):
        pass


class ContLinearScaling(BaseScaling):

    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)
        self.nbr_left = None
        self.nbr_right = None

        self.child_left = None
        self.child_mid = None
        self.child_right = None

    def refine_mid(self):
        if self.child_mid: return self.child_mid
        l, n = self.labda

        # Calculate the support of the refined hat.
        for elem in self.support:
            elem.bisect()
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

    def mass(self):
        result = []
        l, n = self.labda
        self_ip = 0
        if n > 0:
            assert self.nbr_left
            result.append((self.nbr_left, 1 / 6 * 2**-l))
            self_ip += 1 / 3 * 2**-l
        if n < 2**l:
            assert self.nbr_right
            result.append((self.nbr_right, 1 / 6 * 2**-l))
            self_ip += 1 / 3 * 2**-l

        result.append((self, self_ip))
        return result

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


class ThreePointWavelet(BaseWavelet):

    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if self.children: return
        phi_left, phi_mid, phi_right = [phi for phi, _ in self.single_scale]
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


class ThreePointBasis(BaseBasis):
    # Create static mother scaling functions.
    mother_scalings = [
        ContLinearScaling((0, 0), None, [Element.mother_element]),
        ContLinearScaling((0, 1), None, [Element.mother_element])
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
