from linear_operator import LinearOperator
from interval import Interval
from fractions import Fraction

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
            {index
             for index in self.indices if index.labda[0] <= level})

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
        child_right = self.__class__(self.level + 1, self.node_index * 2 + 1, self)
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
        self.support = support # Support is a list.
        self.multi_scale = [] # Tranpose of the wavelet to multiscale.

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

class HaarScaling(BaseScaling):
    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)
        self.children = []

    def refine(self):
        if self.children: return
        self.support[0].bisect()
        l, n = self.labda
        self.children.append(HaarScaling((l+1, 2*n), self, [self.support[0].children[0]]))
        self.children.append(HaarScaling((l+1, 2*n+1), self, [self.support[0].children[1]]))

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

    def eval(self, x, deriv=False):
        result = 0
        for phi, coeff_ss in self.single_scale:
            result += coeff_ss * phi.eval(x, deriv)
        return result

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, *self.labda)

class HaarWavelet(BaseWavelet):
    def __init__(self, labda, parents, single_scale):
        super().__init__(labda, parents, single_scale)

    def refine(self):
        if self.children: return
        phi_left, phi_right = [phi for phi, _ in self.single_scale]
        phi_left.refine()
        phi_right.refine()

        l, n = self.labda
        child_left = HaarWavelet((l+1, 2*n), self, [(phi_left.children[0], 1), (phi_left.children[1], -1)])
        child_right = HaarWavelet((l+1, 2*n+1), self, [(phi_right.children[0], 1), (phi_right.children[1], -1)])
        self.children = [child_left, child_right]

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

        Delta = basis.mother_scalings + list({phi for psi in Lambda for phi, _ in psi.single_scale})
        Lambda = basis.mother_scalings + Lambda
        return basis, MultiscaleIndices(Lambda), MultiscaleIndices(Delta)

    @classmethod
    def origin_refined_basis(cls, max_level):
        assert max_level >= 1
        basis = cls()

        n_wavelets = len(basis.mother_wavelets)
        assert n_wavelets == 1

        Lambda = [] + basis.mother_wavelets
        Lambda_l = Lambda
        for _ in range(max_level - 1):
            parent = Lambda[-1]
            parent.refine()
            Lambda.append(parent.children[0])

        Delta = basis.mother_scalings + list({phi for psi in Lambda for phi, _ in psi.single_scale})
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

class HaarBasis(BaseBasis):
    # Create static mother scaling function.
    mother_scaling = HaarScaling((0, 0), None, [Element.mother_element])
    mother_scalings = [mother_scaling]
    mother_scaling.refine()

    # Create static mother wavelet function.
    mother_wavelet = HaarWavelet((1, 0), None, [(mother_scaling.children[0], 1), (mother_scaling.children[1], -1)])
    mother_wavelets = [mother_wavelet]

    def __init__(self):
        pass

class ThreePointScaling(BaseScaling):
    def __init__(self, labda, parents, support):
        super().__init__(labda, parents, support)
        self.nbr_left = None
        self.nbr_right = None

        self.child_left = None
        self.child_mid  = None
        self.child_right= None

    def refine_mid(self):
        if self.child_mid: return self.child_mid
        l, n = self.labda

        # Calculate the support of the refined hat.
        for elem in self.support: elem.bisect()
        child_support = []
        if n > 0: child_support.append(self.support[0].children[1])
        if n < 2**l: child_support.append(self.support[-1].children[0])

        # Create element.
        child = ThreePointScaling((l+1, 2*n), [self], child_support)
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
        phi_left  = self.nbr_left
        phi_right = self

        # Create child element.
        child = ThreePointScaling((l+1, n*2-1), [phi_left, phi_right], phi_right.support[0].children)
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
        assert self.nbr_right
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
            result.append((self.nbr_left, 1/6 * 2**-l))
            self_ip += 1/3 * 2**-l
        if n < 2**l:
            assert self.nbr_right
            result.append((self.nbr_right, 1/6 * 2**-l))
            self_ip += 1/3 * 2**-l

        result.append((self, self_ip))
        return result

    @staticmethod
    def eval_mother(x, deriv=False):
        """ Evaluates the hat function on [-1,1] centered at 0. """
        left_mask = (-1 < x) & (x <= 0)
        right_mask = (0 < x) & (x < 1)

        if not deriv:
            return left_mask * (1+x) + right_mask * (1-x)
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
        scaling = 2**((l+1)/2)

        # First refine the left part
        phi_children = phi_left.refine_mid(), phi_mid.refine_left(), phi_mid.refine_mid()
        if n == 0:
            child_left = ThreePointWavelet((l+1, 2*n), self, zip(phi_children, (-1*scaling, 1*scaling, -1/2*scaling)))
        else:
            child_left = ThreePointWavelet((l+1, 2*n), self, zip(phi_children, (-1/2*scaling, 1*scaling, -1/2*scaling)))

        # Now refine the right part
        phi_children = phi_mid.refine_mid(), phi_mid.refine_right(), phi_right.refine_mid()
        if  n == 2**(l-1) - 1:
            child_right = ThreePointWavelet((l+1, 2*n+1), self, zip(phi_children, (-1/2* scaling, 1 * scaling, -1 * scaling)))
        else:
            child_right = ThreePointWavelet((l+1, 2*n+1), self, zip(phi_children, (-1/2* scaling, 1 * scaling, -1/2 * scaling)))

        self.children = [child_left, child_right]

class ThreePointBasis(BaseBasis):
    # Create static mother scaling functions.
    mother_scalings = [ThreePointScaling((0, 0), None, [Element.mother_element]),ThreePointScaling((0, 1), None, [Element.mother_element])]
    mother_scalings[0].nbr_right = mother_scalings[1]
    mother_scalings[1].nbr_left = mother_scalings[0]

    # Create the static mother wavelet.
    mother_wavelets = [ThreePointWavelet((1, 0), None, [(mother_scalings[0].refine_mid(), -2**(1/2)), (mother_scalings[0].refine_right(), 2**(1/2)), (mother_scalings[1].refine_mid(), -2**(1/2))])]

    def __init__(self):
        pass
