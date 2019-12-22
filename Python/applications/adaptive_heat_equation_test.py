import numpy as np

from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis
from .adaptive_heat_equation import AdaptiveHeatEquation
from .heat_equation import HeatEquation
from .heat_equation_test import example_rhs_functional


def test_heat_error_reduction(theta=0.7,
                              results_file=None,
                              rhs_factory=example_rhs_functional):
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    triang = InitialTriangulation.unit_square(initial_refinement=1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()

    # Create X^\delta as a sparse grid.
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)

    # Create rhs functionals
    g_functional, u0_functional = rhs_factory(HeatEquation(X_delta))

    # Create adaptive heat equation object.
    adaptive_heat_eq = AdaptiveHeatEquation(X_init=X_delta,
                                            g_functional=g_functional,
                                            u0_functional=u0_functional,
                                            theta=theta)
    results = []
    prev_u_dd_d = None
    while True:
        u_dd_d, residual, info = adaptive_heat_eq.solve_step(x0=prev_u_dd_d,
                                                             solver='pcg')
        prev_u_dd_d = u_dd_d
        info.update({'u_delta': u_dd_d.to_array(), 'res': residual.to_array()})
        results.append(info)
        print(results[-1])
        if results_file is not None:
            import pickle
            pickle.dump(results, open(results_file, 'wb'))


if __name__ == "__main__":
    # test_preconditioned_eigenvalues(max_level=16, sparse_grid=True)
    test_heat_error_reduction(results_file='smooth_solution_adaptive.pkl')
