from applicator import Applicator

from haar_basis import HaarBasis
from basis import support_to_interval
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis
from index_set import MultiscaleIndexSet
from indexed_vector import IndexedVector

from scipy.integrate import quad
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import time
import numpy as np
import cProfile
import pytest


class ScipyLinearOperator(linalg.LinearOperator):
    """ Helper class to interface with scipy.sparse.linalg.{gmres,cg}. """

    def __init__(self, indices, boundary_condition, applicator):
        self.indices = indices
        self.boundary_condition = boundary_condition
        self.applicator = applicator
        self.shape = (len(self.indices), len(self.indices))
        self.dtype = np.dtype('float')

    def _matvec(self, array):
        indexed_vector = IndexedVector(self.indices, array)
        if isinstance(self.applicator, Applicator):
            output = self.applicator.apply(indexed_vector)
        else:
            output = self.applicator.matvec(indexed_vector, self.indices, self.indices)
        for labda in self.boundary_condition:
            output[labda] = 0.0
        return output.asarray(self.indices)


def test_singlescale_ode_solve(plot=False):
    """ Solve a Poisson problem -u'' = f using singlescale stiffnes matrix.

    This also works (upon making the correct changes) for the eqn u' = f
    (using damping matrix). I suppose it also works for u = f using mass matrix.
    """
    prev_L2_error = 1.0
    for ml in range(1, 6):
        basis = ThreePointBasis.uniform_basis(max_level=ml)
        basis.vanish_at_boundary = True
        indices = basis.scaling_indices_on_level(ml)
        boundary_condition = {(ml, 0), (ml, 2**ml)}

        u = lambda t: np.sin(2 * np.pi * t)
        dtt_u = lambda t: -4 * np.pi**2 * np.sin(2 * np.pi * t)

        # Build RHS, simply through quadrature.
        rhs = {}
        for labda in indices:
            supp = support_to_interval(basis.scaling_support(labda))
            rhs[labda] = 0.0 if labda in boundary_condition else quad(
                lambda t: -dtt_u(t) * basis.eval_scaling(labda, t),
                supp.a,
                supp.b,
                points=[float(supp.a), float(supp.mid), float(supp.b), 1.0])[0]
        rhs = IndexedVector(rhs)

        # Build the operator. NB: stiffness matrix is pos-def so we can use CG.
        operator = basis.scaling_stiffness()
        scipy_op = ScipyLinearOperator(indices, boundary_condition, operator)
        sol = IndexedVector(
            indices,
            linalg.cg(scipy_op, rhs.asarray(), atol='legacy')[0])

        # Build our solution.
        u_delta = lambda t: sum(
            sol[labda] * basis.eval_scaling(labda, t) for labda in indices)
        L2_error = quad(lambda t: (u(t) - u_delta(t))**2, 0, 1)[0]
        print("level %s: singlescale stiffness solve L2-error %s" %
              (ml, L2_error))
        assert L2_error < prev_L2_error
        prev_L2_error = L2_error
        if plot:
            tt = np.linspace(0, 1, 1025)
            plt.title("Singlescale stiffness solve on level %s" % ml)
            plt.plot(tt, u_delta(tt), label=r"$u_\delta$")
            plt.plot(tt, u(tt), label=r"$u$")
            plt.legend()
            plt.show()


def test_multiscale_ode_solve(plot=False):
    """ Test solving u = f (zeroth-order ODE) using Mass @ v = <u, v>_{L_2}.

    I was unable to find exactly *which* multiscale boundary conditions I should
    impose in order to get a solution that made any kind of sense, because
    simply setting `boundary_condition` to all labda for which
        psi_lambda(0) != 0 or psi_lambda(1) != 0
    yields weirdness. I don't know -- maybe we can't use this multiscale basis
    directly for solving ODEs?
    """
    prev_L2_error = 1.0
    for ml in range(1, 7):
        basis = ThreePointBasis.uniform_basis(max_level=ml)
        basis.vanish_at_boundary = True
        indices = basis.indices
        boundary_condition = {(0, 0), (0, 1)}
        u = lambda t: np.sin(2 * np.pi * t)
        dtt_u = lambda t: -4 * np.pi**2 * np.sin(2 * np.pi * t)

        # Build RHS.
        rhs_dict = {}
        for labda in indices:
            supp = support_to_interval(basis.wavelet_support(labda))
            rhs_dict[labda] = 0.0 if labda in boundary_condition else quad(
                lambda t: -dtt_u(t) * basis.eval_wavelet(labda, t),
                supp.a,
                supp.b,
                points=[float(supp.a), float(supp.mid), float(supp.b), 1.0])[0]
        rhs = IndexedVector(rhs_dict)

        # Build operator.
        operator = basis.scaling_stiffness()
        applicator = Applicator(basis, operator, indices)
        scipy_op = ScipyLinearOperator(indices, boundary_condition, applicator)
        sol = IndexedVector(
            indices,
            linalg.cg(scipy_op, rhs.asarray(), atol='legacy')[0])

        u_delta = lambda t: sum(
            sol[labda] * basis.eval_wavelet(labda, t) for labda in indices)
        L2_error = quad(lambda t: (u(t) - u_delta(t))**2, 0, 1)[0]
        print("level %s: singlescale mass solve L2-error %s" % (ml, L2_error))
        assert L2_error < prev_L2_error
        prev_L2_error = L2_error
        if plot:
            tt = np.linspace(0, 1, 1025)
            plt.title("Multiscale stiffness matrix solve on level %s" % ml)
            plt.plot(tt, u_delta(tt), label=r"$u_\delta$")
            plt.plot(tt, u(tt), label=r"$u$")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    test_singlescale_ode_solve(plot=True)
    test_multiscale_ode_solve(plot=True)
