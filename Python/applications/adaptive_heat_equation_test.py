import os
from pprint import pprint

import numpy as np
import psutil

from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis
from .adaptive_heat_equation import AdaptiveHeatEquation
from .heat_equation import HeatEquation
from .heat_equation_test import example_rhs_functional


def singular_rhs_functional(heat_eq):
    g = [(
        lambda t: 0,
        lambda xy: 0,
    )]
    g_order = [(0, 0)]
    u0 = [lambda xy: 1]
    u0_order = [1]

    return heat_eq.calculate_rhs_functionals_quadrature(g=g,
                                                        g_order=g_order,
                                                        u0=u0,
                                                        u0_order=u0_order)


def test_heat_error_reduction(theta=0.7,
                              results_file=None,
                              rhs_factory=example_rhs_functional,
                              solver_tol='1e-7'):
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    triang = InitialTriangulation.l_shape(initial_refinement=1)
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
    info = {
        'theta': adaptive_heat_eq.theta,
        'solver_tol': solver_tol,
        'step_info': [],
        'sol_info': [],
    }
    u_dd_d = None
    while True:
        step_info = {}
        # Calculate a new solution.
        u_dd_d, solve_info = adaptive_heat_eq.solve_step(x0=u_dd_d,
                                                         solver='pcg',
                                                         tol=solver_tol)
        step_info.update(solve_info)

        # Store X_delta(_underscore) using centers (in bfs kron order).
        # Also store flattened versions of u_delta and the residual.
        sol_info = {
            'X_delta': [(n.nodes[0].center(), n.nodes[1].center())
                        for n in adaptive_heat_eq.X_delta.bfs_kron()],
            'X_delta_underscore': [(n.nodes[0].center(), n.nodes[1].center())
                                   for n in adaptive_heat_eq.X_dd.bfs_kron()],
            'u_delta':
            u_dd_d.to_array(),
        }

        # Mark and refine.
        residual, mark_info = adaptive_heat_eq.mark_refine(u_dd_d=u_dd_d)
        step_info.update(mark_info)
        sol_info['residual'] = residual.to_array()

        # Store total memory consumption.
        process = psutil.Process(os.getpid())
        step_info['memory'] = process.memory_info().rss

        # Debug.
        print('\n\nstep_info')
        pprint(step_info)
        info['step_info'].append(step_info)
        info['sol_info'].append(sol_info)
        if results_file is not None:
            import pickle
            pickle.dump(info, open(results_file, 'wb'))


if __name__ == "__main__":
    # test_preconditioned_eigenvalues(max_level=16, sparse_grid=True)
    test_heat_error_reduction(
        results_file='singular_solution_adaptive_lshape.pkl',
        rhs_factory=singular_rhs_functional)
