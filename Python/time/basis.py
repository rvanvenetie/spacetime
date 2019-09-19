from fractions import Fraction

from ..datastructures.tree import BinaryNodeAbstract
from .linear_operator import LinearOperator


class Element1D(BinaryNodeAbstract):
    """ Represents an element (interval) as result of dyadic refinement.

    The element (l, n) is an interval on level l given by: [2^-l*n, 2^-l*(n+1)].
    """
    def __init__(self, level, node_index, parent=None):
        super().__init__(parent=parent, children=None)

        self.level = level
        self.node_index = node_index

        # Create variables to register non-zero scaling functions.
        # The ordering of functions inside these lists is important.
        # TODO: Is this the right place?
        self.phi_disc_const = None
        self.phi_disc_lin = [None, None]
        self.phi_cont_lin = [None, None]

        # Add some extra variables necessary for applicator.
        # TODO: Is this the right place?
        self.Lambda_in = False
        self.Lambda_out = False
        self.Pi_in = False
        self.Pi_out = False

    def refine(self):
        if not self.children:
            child_left = self.__class__(self.level + 1, self.node_index * 2,
                                        self)
            child_right = self.__class__(self.level + 1,
                                         self.node_index * 2 + 1, self)
            self.children = [child_left, child_right]
        return self.children

    @property
    def interval(self):
        h = Fraction(1, 2**self.level)
        return (h * self.node_index, h * (self.node_index + 1))

    def __repr__(self):
        return 'Element1D({}, {})'.format(self.level, self.node_index)


class Function:
    """ This is a base class for an object represention a basis function.  """
    def __init__(self, labda):
        self.labda = labda

        # TODO: This should be removed, or neatly integrated.
        self.reset_coeff()

    def reset_coeff(self):
        """ Resets the coefficients stored in this function object. """
        self.coeff = [0] * 3

    def eval(self, x, deriv=False):
        pass

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, *self.labda)


class Scaling(Function):
    def __init__(self, labda, parents, support):
        super().__init__(labda)
        self.parents = parents
        self.support = support  # Support is a list.
        self.multi_scale = []  # Transpose of the wavelet to multiscale.

    def prolongate(self):
        """ Returns a list of pairs with the corresponding coefficients. """
        pass

    def restrict(self):
        """ The adjoint of transposing. """
        pass


class Wavelet(Function):
    def __init__(self, labda, parents, single_scale):
        super().__init__(labda)
        self.parents = parents
        self.single_scale = list(single_scale)
        self.children = []
        self.support = []
        for phi, coeff in self.single_scale:
            self.support.extend(phi.support)
            # Register this wavelet in the corresponding phi.
            phi.multi_scale.append((self, coeff))

    def eval(self, x, deriv=False):
        result = 0
        for phi, coeff_ss in self.single_scale:
            result += coeff_ss * phi.eval(x, deriv)
        return result


class MultiscaleFunctions:
    """ Immutable set of multiscale functions.  """
    def __init__(self, functions):
        self.functions = functions
        self.maximum_level = max([fn.labda[0] for fn in functions])
        self.per_level = [[] for _ in range(self.maximum_level + 1)]
        for fn in functions:
            self.per_level[fn.labda[0]].append(fn)

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
        Lambda = [] + basis.mother_wavelets
        Lambda_l = Lambda
        for _ in range(max_level - 1):
            Lambda_new = []
            for psi in Lambda_l:
                psi.refine()
                Lambda_new.extend(psi.children)
            Lambda_l = Lambda_new
            Lambda.extend(Lambda_l)

        Lambda = basis.mother_scalings + Lambda
        return basis, MultiscaleFunctions(Lambda)

    @classmethod
    def origin_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        n_wavelets = len(basis.mother_wavelets)
        Lambda = basis.mother_scalings + basis.mother_wavelets
        for _ in range(max_level - 1):
            parents = list(Lambda[-n_wavelets:])
            for parent in parents:
                parent.refine()
                #TODO: This assumes the left child is always 0
                Lambda.append(parent.children[0])

        return basis, MultiscaleFunctions(Lambda)

    @classmethod
    def end_points_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        n_wavelets = len(basis.mother_wavelets)

        # Make a copy of the mother_wavelets list.
        Lambda = [] + basis.mother_wavelets

        # First add all the wavelets near the origin
        # TODO: assumets the left child is 0
        for _ in range(max_level - 1):
            parents = list(Lambda[-n_wavelets:])
            for parent in parents:
                parent.refine()
                Lambda.append(parent.children[0])

        # Now add all the wavelets near the endpoint.
        # TODO: dirty hacks involved here
        Lambda = Lambda[n_wavelets:] + basis.mother_wavelets
        for _ in range(max_level - 1):
            parents = list(Lambda[-n_wavelets:])
            for parent in parents:
                parent.refine()
                Lambda.append(parent.children[-1])

        Lambda = basis.mother_scalings + Lambda
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
