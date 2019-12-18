import scipy.sparse.linalg as spla

from ..datastructures.applicator import (BlockApplicator,
                                         LinearOperatorApplicator,
                                         CompositeApplicator)
from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.multi_tree_function import DoubleTreeFunction
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..space import applicator as s_applicator
from ..space import operators as s_operators
from ..spacetime.applicator import Applicator, BlockDiagonalApplicator
from ..spacetime.basis import generate_y_delta
from ..time import applicator as t_applicator
from ..time import operators as t_operators
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis


class HeatEquation:
    """ Simple solution class. """
    def __init__(self,
                 X_delta,
                 Y_delta=None,
                 dirichlet_boundary=True,
                 solver='minres'):
        if Y_delta is None:
            Y_delta = generate_y_delta(X_delta)

        self.X_delta = X_delta
        self.Y_delta = Y_delta
        self.dirichlet_boundary = dirichlet_boundary
        self.solver = solver

        print('HeatEquation with #(Y_delta, X_delta)=({}, {})'.format(
            len(Y_delta.bfs()), len(X_delta.bfs())))

        time_basis_X = ThreePointBasis()
        time_basis_Y = OrthonormalBasis()

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
            applicator_time=applicator_time(t_operators.mass, time_basis_Y),
            applicator_space=applicator_space(s_operators.StiffnessOperator))
        B_1 = Applicator(
            Lambda_in=X_delta,
            Lambda_out=Y_delta,
            applicator_time=applicator_time(t_operators.transport,
                                            time_basis_X, time_basis_Y),
            applicator_space=applicator_space(s_operators.MassOperator))
        B_2 = Applicator(
            Lambda_in=X_delta,
            Lambda_out=Y_delta,
            applicator_time=applicator_time(t_operators.mass, time_basis_X,
                                            time_basis_Y),
            applicator_space=applicator_space(s_operators.StiffnessOperator))
        self.B = B_1 + B_2
        self.BT = self.B.transpose()
        self.m_gamma = Applicator(
            Lambda_in=X_delta,
            Lambda_out=X_delta,
            applicator_time=applicator_time(t_operators.trace, time_basis_X),
            applicator_space=applicator_space(s_operators.MassOperator))

        if solver == 'minres':
            self.mat = BlockApplicator([[self.A_s, self.B],
                                        [self.BT, -self.m_gamma]])
        elif solver in ['cg-schur', 'pcg-schur']:
            self.P_Y = BlockDiagonalApplicator(
                Y_delta,
                applicator_space=applicator_space(
                    s_operators.DirectInverse,
                    forward_cls=s_operators.StiffnessOperator))
            self.mat = CompositeApplicator([self.B, self.P_Y, self.BT
                                            ]) + self.m_gamma
            if solver == 'pcg-schur':
                self.P_X = BlockDiagonalApplicator(
                    X_delta,
                    applicator_space=applicator_space(
                        s_operators.XPreconditioner,
                        precond_cls=s_operators.DirectInverse))
        else:
            raise NotImplementedError("Unknown method " % solver)
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
                      mlt_tree_cls=DoubleTreeVector,
                      both_spaces=False):
        if self.solver == 'minres' or both_spaces:
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

    def calculate_rhs_vector(self, g, g_order, u0, u0_order):
        """ Generates a rhs vector for the given rhs g and initial cond u_0 .

        This assumes that g is given as a sum of seperable functions.

        Args:
          g: the rhs of the heat equation. Given as list of tuples,
            where each tuple (f_t, f_xy) represents the tensor product f_tf_xy.
          g_order: a tuple describing the time/space polynomial degree of g.
          u0: a function of space that represents the initial condition.
          u0_order: the degree of u0.
        """
        def call_quad_g(nv, _):
            """ Helper function to do the quadrature for the rhs g. """
            if nv.is_metaroot(): return
            nv.value = sum(nv.nodes[0].inner_quad(g0, g_order=g_order[0]) *
                           nv.nodes[1].inner_quad(g1, g_order=g_order[1])
                           for g0, g1 in g)

        def call_quad_u0(nv, _):
            """ Helper function to do the quadrature for the rhs u0. """
            if nv.is_metaroot(): return
            nv.value = -nv.nodes[0].eval(0) * nv.nodes[1].inner_quad(
                u0, g_order=u0_order)

        if self.solver == 'minres':
            return self.create_vector((call_quad_g, call_quad_u0))
        else:
            rhs = self.create_vector((call_quad_g, call_quad_u0),
                                     both_spaces=True)
            return self.BT.apply(self.P_Y.apply(rhs[0])).axpy(rhs[1], -1.0)

    def solve(self, rhs, iter_callback=None):
        num_iters = 0

        def call_iterations(vec):
            nonlocal num_iters
            if iter_callback: iter_callback(vec)
            print(".", end='', flush=True)
            num_iters += 1

        if self.solver == "minres":
            solver = spla.minres
        elif self.solver == "cg-schur":
            solver = spla.cg
        elif self.solver == "pcg-schur":
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
