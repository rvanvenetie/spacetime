from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache

import numpy as np
import quadpy

from ..datastructures.function import FunctionInterface
from ..datastructures.tree import BinaryNodeAbstract, MetaRoot, NodeAbstract
from ..datastructures.tree_view import NodeViewInterface
from .linear_operator import LinearOperator
from .sparse_vector import SparseVector


@lru_cache(maxsize=10)
def _get_quadrature_scheme(order):
    # order == 2 * n - 1.
    return quadpy.line_segment.gauss_legendre((order + 2) // 2)


class Element1D(BinaryNodeAbstract):
    """ Represents an element (interval) as result of dyadic refinement.

    The element (l, n) is an interval on level l given by: [2^-l*n, 2^-l*(n+1)].
    """
    __slots__ = [
        'level',
        'left_node_idx',
        'phi_disc_const',
        'phi_disc_lin',
        'phi_cont_lin',
        'psi_ortho',
        'Lambda_in',
        'Lambda_out',
        'Pi_in',
        'Pi_out',
        'Sigma_psi_out',
        'Theta_psi_in',
    ]

    def __init__(self, level, left_node_idx, parent=None):
        super().__init__(parent=parent, children=None)
        if parent: assert parent.level + 1 == level

        self.level = level
        self.left_node_idx = left_node_idx

        # Create variables to register non-zero scaling functions.
        # The ordering of functions inside these lists is important.
        # TODO: Is this the right place?
        self.phi_disc_const = None
        self.phi_disc_lin = [None, None]
        self.phi_cont_lin = [None, None]

        # Create variables to register non-zero wavelet functions.
        # The ordering of functions inside these lists is important.
        # TODO: Is this the right place?
        self.psi_ortho = [None, None]

        # Add some extra variables necessary for time applicator.
        # TODO: Is this the right place?
        self.Lambda_in = False
        self.Lambda_out = False
        self.Pi_in = False
        self.Pi_out = False

        # Add some extra variables the spacetime applicator
        # TODO: Is this the right place?
        self.Sigma_psi_out = []
        self.Theta_psi_in = False

    def _refine_psi_orthonormal(self):
        """ Ensures that Orthonormal wavelets are all set for this element.

        This requires that the mother element has valid references.
        """
        if not all(self.psi_ortho):
            assert self.level > 0

            # Recursively find/create the wavelets of the parents.
            parent_ortho = self.parent._refine_psi_orthonormal()

            # Refine the parent wavelets, so this element gets its children.
            parent_ortho[0].refine()
            assert all(self.psi_ortho)

        return self.psi_ortho

    def _refine_phi_cont_lin(self):
        """ Ensures that all cont lin scaling functions exist on this elem. """
        if not all(self.phi_cont_lin):
            # Ensure that the parent has both scalar functions.
            assert self.level > 0 and all(self.parent.phi_cont_lin)

            # We are the left child element.
            if self.left_node_idx % 2 == 0:
                if not self.phi_cont_lin[0]:
                    self.parent.phi_cont_lin[0].refine_mid()
                if not self.phi_cont_lin[1]:
                    self.parent.phi_cont_lin[0].refine_right()

            # We are the right child element.
            else:
                if not self.phi_cont_lin[0]:
                    self.parent.phi_cont_lin[1].refine_left()
                if not self.phi_cont_lin[1]:
                    self.parent.phi_cont_lin[1].refine_mid()

            assert all(self.phi_cont_lin)

        return self.phi_cont_lin

    def refine(self):
        if not self.children:
            child_left = Element1D(self.level + 1, self.left_node_idx * 2,
                                   self)
            child_right = Element1D(self.level + 1, self.left_node_idx * 2 + 1,
                                    self)
            self.children = [child_left, child_right]
        return self.children

    @property
    def interval(self):
        h = Fraction(1, 2**self.level)
        return (h * self.left_node_idx, h * (self.left_node_idx + 1))

    def __repr__(self):
        return 'Element1D({}, {})'.format(self.level, self.left_node_idx)


class CoefficientFunction1D(NodeAbstract, FunctionInterface):
    """ This is a base represention of a basis function with coefficients. """
    __slots__ = ['labda', 'support', 'coeff']

    def __init__(self, labda, support, parents=None):
        super().__init__(parents=parents, children=None)
        self.labda = labda
        self.support = support  # Support is a list of Element1D's.

        # TODO: This should be removed, or neatly integrated.
        self.reset_coeff()

    @property
    def level(self):
        return self.labda[0]

    @property
    def index(self):
        return self.labda[1]

    @property
    def interval(self):
        return (self.support[0].interval[0], self.support[-1].interval[1])

    def reset_coeff(self):
        """ Resets the coefficients stored in this function object. """
        self.coeff = [0, 0]

    def center(self):
        return (sum(self.interval) / 2, self.level)

    def support_contains(self, t):
        a, b = self.interval()
        return float(a) <= t <= float(b)

    def inner_quad(self, g, g_order=2, deriv=False):
        """ Computes <g, self> or <g, d/dt self> by quadrature. """
        func = lambda t: (self.eval(t, deriv) * g(t))
        scheme = _get_quadrature_scheme(g_order + self.order)
        result = 0.0
        for elem in self.support:
            interval = list(map(float, elem.interval))
            result += scheme.integrate(func, interval)
        return result

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, *self.labda)


class Scaling(CoefficientFunction1D):
    __slots__ = ['multi_scale']

    def __init__(self, labda, support, parents=None):
        super().__init__(labda=labda, support=support, parents=parents)
        self.multi_scale = []  # Transpose of the wavelet to multiscale.

    def prolongate(self):
        """ Returns a list of pairs with the corresponding coefficients. """

    def restrict(self):
        """ The adjoint of transposing. """


class Wavelet(CoefficientFunction1D):
    __slots__ = ['single_scale']

    def __init__(self, labda, single_scale, parents=None):
        super().__init__(labda, support=[], parents=parents)

        assert single_scale
        self.single_scale = list(single_scale)
        support = []
        for phi, coeff in self.single_scale:
            support.extend(phi.support)
            # Register this wavelet in the corresponding phi.
            phi.multi_scale.append((self, coeff))

        # Deduplicate the support while keeping the ordering intact.
        self.support = list(OrderedDict.fromkeys(support))

    def eval(self, t, deriv=False):
        if not isinstance(t, np.ndarray): t = np.array([float(t)])
        assert len(t.shape) == 1 or t.shape[0] == 1
        result = np.zeros(t.shape)
        for phi, coeff_ss in self.single_scale:
            result += coeff_ss * phi.eval(t, deriv)
        return result if t.shape[0] > 1 else result[0]


class MultiscaleFunctions:
    """ Immutable set of multiscale functions.  """
    def __init__(self, functions):
        if isinstance(functions, MetaRoot):
            functions = [nv for nv in functions.bfs()]
        if isinstance(functions, NodeViewInterface):
            functions = [nv.node for nv in functions.bfs()]
        if isinstance(functions, SparseVector):
            functions = list(functions)
        self.functions = functions
        self.maximum_level = max([-1] + [fn.level for fn in functions])
        self.per_level = [[] for _ in range(self.maximum_level + 1)]
        for fn in functions:
            self.per_level[fn.level].append(fn)

    def __repr__(self):
        return r"MSIS(%s)" % self.functions

    def __len__(self):
        return len(self.functions)

    def __iter__(self):
        return self.functions.__iter__()

    def __next__(self):
        return self.functions.__next__()

    def __contains__(self, index):
        return self.functions.__contains__(index)

    def on_level(self, level):
        if level > self.maximum_level: return []
        return self.per_level[level]

    def until_level(self, level):
        """ Expensive (but linear in size) method. Mainly for testing. """
        return MultiscaleFunctions(
            {index
             for index in self.functions if index.labda[0] <= level})

    def single_scale_functions(self):
        """ Expensive. Returns a list of all single scale functions.

        Assumes that this set represents wavelets.
        """
        Delta = []
        for psi in self.functions:
            if isinstance(psi, Wavelet):
                Delta.extend([phi for phi, _ in psi.single_scale])
            else:
                assert isinstance(psi, Scaling)
                Delta.append(psi)
        return MultiscaleFunctions(set(Delta))


class Basis:
    @classmethod
    def uniform_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        # Start refining all wavelets. Copy the mother_wavelets list.
        basis.metaroot_wavelet.uniform_refine(max_level)
        Lambda = [
            psi for psi in basis.metaroot_wavelet.bfs()
            if psi.level <= max_level
        ]
        return basis, MultiscaleFunctions(Lambda)

    @classmethod
    def origin_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        # Ensure all wavelets are available.
        basis.metaroot_wavelet.uniform_refine(1)

        # Find all wavelets up to level 1.
        Lambda = [
            psi for psi in basis.metaroot_wavelet.bfs() if psi.level <= 1
        ]
        n_wavelets = len([psi for psi in Lambda if psi.level == 1])
        for _ in range(max_level - 1):
            # Get the left-most parent
            parent = Lambda[-n_wavelets]

            # Assume that its children are all wavelets in the corner.
            parent.refine()
            assert len(parent.children) >= n_wavelets
            Lambda.extend(parent.children[:n_wavelets])

        return basis, MultiscaleFunctions(Lambda)

    @classmethod
    def end_points_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        # Ensure all wavelets are available.
        basis.metaroot_wavelet.uniform_refine(1)

        # Find all wavelets at level 1.
        mother_wavelets = [
            psi for psi in basis.metaroot_wavelet.bfs() if psi.level == 1
        ]
        Lambda = mother_wavelets.copy()
        n_wavelets = len(mother_wavelets)

        # First add all the wavelets near the origin
        for _ in range(max_level - 1):
            # Get the left-most parent
            parent = Lambda[-n_wavelets]

            # Assume that its children are all wavelets in the corner.
            parent.refine()
            assert len(parent.children) >= n_wavelets
            Lambda.extend(parent.children[:n_wavelets])

        # Now add all the wavelets near the endpoint.
        # TODO: dirty hacks involved here
        Lambda = Lambda[n_wavelets:] + mother_wavelets
        for _ in range(max_level - 1):
            # Get rightmost parent
            parent = Lambda[-1]
            # Assume that its children are all wavelets in the corner.
            parent.refine()
            assert len(parent.children) >= n_wavelets
            Lambda.extend(parent.children[-n_wavelets:])

        Lambda = basis.metaroot_wavelet.children + Lambda
        return basis, MultiscaleFunctions(Lambda)

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


# Initializes a mother_element to be used for all bases.
mother_element = Element1D(0, 0, None)
element_meta_root = MetaRoot(mother_element)
