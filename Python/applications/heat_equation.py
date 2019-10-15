import scipy

from .. import space, time
from ..datastructures.applicator import (BlockApplicator,
                                         LinearOperatorApplicator)
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector)
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..space import applicator
from ..space import operators as s_operators
from ..spacetime.applicator import Applicator
from ..spacetime.basis import generate_y_delta
from ..time import applicator
from ..time import operators as t_operators
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis


class HeatEquation:
    """ Simple solution class. """
    def __init__(self, X_delta, Y_delta=None, dirichlet_boundary=True):
        if Y_delta is None:
            Y_delta = generate_y_delta(X_delta)

        self.X_delta = X_delta
        self.Y_delta = Y_delta
        self.dirichlet_boundary = dirichlet_boundary

        print('HeatEquation with #(Y_delta, X_delta)=({}, {})'.format(
            len(X_delta.bfs()), len(Y_delta.bfs())))

        time_basis_X = ThreePointBasis()
        time_basis_Y = OrthonormalBasis()

        def applicator_time(operator, basis_in, basis_out=None):
            """ Helper function to generate a time applicator. """
            if basis_out is None: basis_out = basis_in
            return time.applicator.Applicator(operator(basis_in, basis_out),
                                              basis_in=basis_in,
                                              basis_out=basis_out)

        def applicator_space(operator):
            """ Helper function to generate a space applicator. """
            return space.applicator.Applicator(
                operator(dirichlet_boundary=dirichlet_boundary))

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
        self.m_gamma = -Applicator(
            Lambda_in=X_delta,
            Lambda_out=X_delta,
            applicator_time=applicator_time(t_operators.trace, time_basis_X),
            applicator_space=applicator_space(s_operators.MassOperator))

        self.mat = BlockApplicator([[self.A_s, self.B],
                                    [self.BT, self.m_gamma]])

        # Also turn this block applicator into a linear operator.
        self.linop = LinearOperatorApplicator(applicator=self.mat,
                                              input_vec=self.create_vector())

    @staticmethod
    def enforce_dirichlet_boundary(vector):
        """ This sets all the dofs belonging to dirichlet conditions to 0. """
        for nv in vector.bfs():
            if nv.nodes[1].on_domain_boundary:
                nv.value = 0

    def create_vector(self, call_postprocess=None):
        if isinstance(call_postprocess, tuple):
            call_postprocess = (call_postprocess, call_postprocess)

        result = BlockTreeVector((
            self.Y_delta.deep_copy(mlt_node_cls=DoubleNodeVector,
                                   mlt_tree_cls=DoubleTreeVector,
                                   call_postprocess=call_postprocess[0]),
            self.X_delta.deep_copy(mlt_node_cls=DoubleNodeVector,
                                   mlt_tree_cls=DoubleTreeVector,
                                   call_postprocess=call_postprocess[1]),
        ))
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
            hbf = HierarchicalBasisFunction(nv.nodes[1])
            nv.value = sum(nv.nodes[0].inner_quad(g0, g_order=g_order[0]) *
                           hbf.inner_quad(g1, order=g_order[1])
                           for g0, g1 in g)

        def call_quad_u0(nv, _):
            """ Helper function to do the quadrature for the rhs u0. """
            hbf = HierarchicalBasisFunction(nv.nodes[1])
            nv.value = nv.nodes[0].eval(0) * hbf.inner_quad(u0, order=u0_order)

        return self.create_vector((call_quad_g, call_quad_u0))
                                       call_postprocess=call_quad_g)

    def solve(self, rhs):
        num_iters = 0

        def call_iterations(_):
            nonlocal num_iters
            num_iters += 1

        rhs_array = rhs.to_array()

        result_array, info = scipy.sparse.linalg.minres(
            self.linop, b=rhs_array, x0=rhs_array, callback=call_iterations)
        assert info == 0

        result_vec = self.create_vector()
        result_vec.from_array(result_array)
        return result_vec, num_iters
