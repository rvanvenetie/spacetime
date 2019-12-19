import scipy.sparse.linalg as spla

from ..datastructures.applicator import (BlockApplicator, CompositeApplicator,
                                         LinearOperatorApplicator)
from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.functional import SumFunctional
from ..datastructures.multi_tree_function import DoubleTreeFunction
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..space import applicator as s_applicator
from ..space import functional as s_functional
from ..space import operators as s_operators
from ..spacetime.applicator import Applicator, BlockDiagonalApplicator
from ..spacetime.basis import generate_y_delta
from ..spacetime.functional import TensorFunctional
from ..time import applicator as t_applicator
from ..time import functional as t_functional
from ..time import operators as t_operators
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis


class HeatEquation:
    """ Simple solution class. """
    def __init__(self,
                 X_delta,
                 Y_delta=None,
                 dirichlet_boundary=True,
                 formulation='saddle'):
        if Y_delta is None:
            Y_delta = generate_y_delta(X_delta)

        self.X_delta = X_delta
        self.Y_delta = Y_delta
        self.dirichlet_boundary = dirichlet_boundary
        self.formulation = formulation

        print('HeatEquation with #(Y_delta, X_delta)=({}, {})'.format(
            len(Y_delta.bfs()), len(X_delta.bfs())))

        self.time_basis_X = ThreePointBasis()
        self.time_basis_Y = OrthonormalBasis()

        def applicator_time(operator, basis_in, basis_out=None):
            """ Helper function to generate a time applicator. """
            if basis_out is None: basis_out = basis_in
            return t_applicator.Applicator(operator(basis_in, basis_out),
                                           basis_in=basis_in,
                                           basis_out=basis_out)

        def applicator_space(operator, **kwargs):
            """ Helper function to generate a space applicator. """
            return s_applicator.Applicator(
                operator(dirichlet_boundary=dirichlet_boundary, **kwargs))

        self.A_s = Applicator(
            Lambda_in=Y_delta,
            Lambda_out=Y_delta,
            applicator_time=applicator_time(t_operators.mass,
                                            self.time_basis_Y),
            applicator_space=applicator_space(s_operators.StiffnessOperator))
        B_1 = Applicator(
            Lambda_in=X_delta,
            Lambda_out=Y_delta,
            applicator_time=applicator_time(t_operators.transport,
                                            self.time_basis_X,
                                            self.time_basis_Y),
            applicator_space=applicator_space(s_operators.MassOperator))
        B_2 = Applicator(
            Lambda_in=X_delta,
            Lambda_out=Y_delta,
            applicator_time=applicator_time(t_operators.mass,
                                            self.time_basis_X,
                                            self.time_basis_Y),
            applicator_space=applicator_space(s_operators.StiffnessOperator))
        self.B = B_1 + B_2
        self.BT = self.B.transpose()
        self.m_gamma = Applicator(
            Lambda_in=X_delta,
            Lambda_out=X_delta,
            applicator_time=applicator_time(t_operators.trace,
                                            self.time_basis_X),
            applicator_space=applicator_space(s_operators.MassOperator))

        if formulation == 'saddle':
            self.mat = BlockApplicator([[self.A_s, self.B],
                                        [self.BT, -self.m_gamma]])
        elif formulation == 'schur':
            self.P_Y = BlockDiagonalApplicator(
                Y_delta,
                applicator_space=applicator_space(
                    s_operators.DirectInverse,
                    forward_op_ctor=s_operators.StiffnessOperator))
            self.P_X = BlockDiagonalApplicator(
                X_delta,
                applicator_space=applicator_space(
                    s_operators.XPreconditioner,
                    precond_cls=s_operators.DirectInverse,
                    alpha=0.35))

            self.mat = CompositeApplicator([self.B, self.P_Y, self.BT
                                            ]) + self.m_gamma
        else:
            raise NotImplementedError("Unknown method " % formulation)
        self.linop = LinearOperatorApplicator(applicator=self.mat,
                                              input_vec=self.create_vector())

    @staticmethod
    def enforce_dirichlet_boundary(vector):
        """ This sets all the dofs belonging to dirichlet conditions to 0. """
        for nv in vector.bfs():
            if nv.nodes[1].on_domain_boundary:
                nv.value = 0

    def create_vector(self,
                      call_postprocess=None,
                      mlt_tree_cls=DoubleTreeVector):
        """ Creates a (empty) solution block vector. """
        if self.formulation == 'saddle':
            if not isinstance(call_postprocess, tuple):
                call_postprocess = (call_postprocess, call_postprocess)
            result = BlockTreeVector(
                (self.Y_delta.deep_copy(mlt_tree_cls=mlt_tree_cls,
                                        call_postprocess=call_postprocess[0]),
                 self.X_delta.deep_copy(mlt_tree_cls=mlt_tree_cls,
                                        call_postprocess=call_postprocess[1])))
        else:
            assert not isinstance(call_postprocess, tuple)
            result = self.X_delta.deep_copy(mlt_tree_cls=mlt_tree_cls,
                                            call_postprocess=call_postprocess)
        if self.dirichlet_boundary:
            self.enforce_dirichlet_boundary(result)

        return result

    def calculate_rhs_vector(self, g_functional, u0_functional):
        """ Generates a rhs vector for a general rhs g and initial cond u_0. """
        rhs_g = g_functional.eval(self.Y_delta)
        rhs_u0 = u0_functional.eval(self.X_delta)
        rhs_u0 *= -1

        # Put the vectors in a block.
        rhs = BlockTreeVector((rhs_g, rhs_u0))

        # Ensure the dirichlet boundary conditions.
        if self.dirichlet_boundary:
            self.enforce_dirichlet_boundary(rhs)

        if self.formulation == 'saddle':
            return rhs
        else:
            f = self.BT.apply(self.P_Y.apply(rhs[0]))
            f -= rhs[1]
            return f

    def calculate_rhs_functionals_quadrature(self, g, g_order, u0, u0_order):
        """ Generates a rhs functional for a seperable rhs g and initial cond u_0 .

        This assumes that g is given as a sum of seperable functions.

        Args:
          g: the rhs of the heat equation. Given as list of tuples,
            where each tuple (f_t, f_xy) represents the tensor product f_tf_xy.
          g_order: a list of tuples describing the
            time/space polynomial degree of g.
          u0: a function of space that represents the initial condition.
          u0_order: the degree of u0.
        """
        assert isinstance(g, list) and isinstance(g_order, list)
        assert isinstance(u0, list) and isinstance(u0_order, list)

        # Create the right hand side g, the spacetime mass inner product.
        g_functionals = []
        for ((g_time, g_space), (g_time_order,
                                 g_space_order)) in zip(g, g_order):
            functional_time = t_functional.Functional(
                t_operators.quadrature(g=g_time, g_order=g_time_order),
                basis=self.time_basis_Y,
            )
            functional_space = s_functional.Functional(
                s_operators.QuadratureFunctional(g=g_space,
                                                 g_order=g_space_order))
            g_functionals.append(
                TensorFunctional(functional_time=functional_time,
                                 functional_space=functional_space))

        g_functional = SumFunctional(functionals=g_functionals)

        # Calculate -gamma_0'u_0, eval in time, mass in space.
        u0_functionals = []
        for u0, u0_order in zip(u0, u0_order):
            functional_time = t_functional.Functional(
                t_operators.evaluation(t=0),
                basis=self.time_basis_X,
            )
            functional_space = s_functional.Functional(
                s_operators.QuadratureFunctional(g=u0, g_order=u0_order))

            u0_functionals.append(
                TensorFunctional(functional_time=functional_time,
                                 functional_space=functional_space))
            print(u0_functionals[-1].eval(self.X_delta).to_array())
        u0_functional = SumFunctional(u0_functionals)
        return g_functional, u0_functional

    def solve(self, rhs, solver=None, iter_callback=None):
        # Set a default value for solver.
        if solver is None:
            solver = {'saddle': 'minres', 'schur': 'cg'}[self.formulation]

        # Check input of solver.
        if self.formulation == 'saddle': assert solver == 'minres'
        if self.formulation == 'schur':
            assert solver in ['cg', 'pcg']

        num_iters = 0

        def call_iterations(vec):
            nonlocal num_iters
            if iter_callback: iter_callback(vec)
            print(".", end='', flush=True)
            num_iters += 1

        if solver == "minres":
            solver = spla.minres
        elif solver == "cg":
            solver = spla.cg
        elif solver == "pcg":
            solver = lambda S, b, callback: spla.cg(
                S,
                b=b,
                M=LinearOperatorApplicator(applicator=self.P_X,
                                           input_vec=self.create_vector()),
                callback=callback)
        else:
            raise NotImplementedError("Unrecognized method '%s'" % self.solver)
        result_array, info = solver(self.linop,
                                    b=rhs.to_array(),
                                    callback=call_iterations)
        result_fn = self.create_vector(mlt_tree_cls=DoubleTreeFunction)
        result_fn.from_array(result_array)
        print('in solve', result_fn)
        print(end='\n')
        assert info == 0
        return result_fn, num_iters

    def time_per_dof(self):
        return self.linop.time_per_dof()
