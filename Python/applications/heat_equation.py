from .. import space, time
from ..datastructures.applicator import BlockApplicator
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
    def __init__(self, X_delta, Y_delta=None):
        if Y_delta is None:
            Y_delta = generate_y_delta(X_delta)

        self.X_delta = X_delta
        self.Y_delta = Y_delta

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
            return space.applicator.Applicator(operator())

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

    def solve(self, rhs):
        pass
