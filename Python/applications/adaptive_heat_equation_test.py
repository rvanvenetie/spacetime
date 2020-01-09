import os
import random
import time
from pprint import pprint

import numpy as np

import psutil

from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis
from .adaptive_heat_equation import AdaptiveHeatEquation
from .error_estimator import AuxiliaryErrorEstimator
from .heat_equation import HeatEquation
from .heat_equation_test import example_rhs_functional, example_u0_data


def test_dorfler_marking():
    class FakeNode:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

    n = 100
    nodes = [FakeNode(random.random()) for _ in range(n)]
    l2_norm = np.sqrt(sum(fn.value**2 for fn in nodes))

    # Test theta == 0.
    assert len(AdaptiveHeatEquation.dorfler_marking(nodes, 0)) == 0

    # Test theta == 1
    assert AdaptiveHeatEquation.dorfler_marking(nodes,
                                                1) == sorted(nodes)[::-1]

    for theta in [0.3, 0.5, 0.7]:
        bulk_nodes = AdaptiveHeatEquation.dorfler_marking(nodes, theta)
        assert len(bulk_nodes) < len(nodes)

        bulk_l2_norm = np.sqrt(sum(fn.value**2 for fn in bulk_nodes))
        assert bulk_l2_norm >= theta * l2_norm

        # Check that shuffled gives same results
        random.shuffle(nodes)
        assert AdaptiveHeatEquation.dorfler_marking(nodes, theta) == bulk_nodes


def test_heat_error_reduction(theta=0.7):
    """ Simple test that applies the adaptive loop for a few iterations. """

    # Create space part.
    triang = InitialTriangulation.unit_square(initial_refinement=1)
    triang.elem_meta_root.uniform_refine(1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()

    # Create X^\delta containing only the roots.
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)

    # Create rhs functionals
    g_functional, u0_functional = example_rhs_functional(HeatEquation(X_delta))

    # Create adaptive heat equation object.
    adaptive_heat_eq = AdaptiveHeatEquation(X_init=X_delta,
                                            g_functional=g_functional,
                                            u0_functional=u0_functional,
                                            theta=theta)

    # Solve.
    sol, info = adaptive_heat_eq.solve(max_iters=2)

    # Some check that seems to hold.
    assert info['errors'][-1] < 0.1


def singular_u0_unit_square_data():
    return (lambda xy: 1, 1, 1.0)


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


def run_adaptive_loop(initial_triangulation='square',
                      theta=0.7,
                      results_file=None,
                      initial_refinement=1,
                      rhs_functional_factory=singular_rhs_functional,
                      u0_data=singular_u0_unit_square_data,
                      solver_tol='1e-7'):
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    if initial_triangulation in ['unit_square', 'square']:
        triang = InitialTriangulation.unit_square(
            initial_refinement=initial_refinement)
    elif initial_triangulation in ['lshape', 'l_shape']:
        triang = InitialTriangulation.l_shape(
            initial_refinement=initial_refinement)
    else:
        assert False

    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()

    # Create X^\delta as a sparse grid.
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)

    # Create rhs functionals
    g_functional, u0_functional = rhs_functional_factory(HeatEquation(X_delta))

    # Create auxiliary error estimator.
    aux_error_estimator = AuxiliaryErrorEstimator(g_functional, u0_functional,
                                                  *u0_data)

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
    time_start = time.time()
    while True:
        time_start_iteration = time.time()
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
            'u_delta':
            u_dd_d.to_array(),
        }

        # Mark and refine.
        residual, mark_info = adaptive_heat_eq.mark_refine(u_dd_d=u_dd_d)
        step_info.update(mark_info)
        sol_info['residual'] = residual.to_array()

        step_info['aux_error'] = aux_error_estimator.estimate(
            u_dd_d,
            adaptive_heat_eq.X_delta,
            adaptive_heat_eq.Y_dd,
            heat_dd_d=adaptive_heat_eq.heat_dd_d)

        # Store total memory consumption.
        process = psutil.Process(os.getpid())
        step_info['memory'] = process.memory_info().rss
        step_info['time_iteration'] = time.time() - time_start_iteration
        step_info['time_since_start'] = time.time() - time_start

        # Debug.
        print('\n\nstep_info')
        pprint(step_info)
        info['step_info'].append(step_info)
        info['sol_info'].append(sol_info)
        if results_file is not None:
            import pickle
            pickle.dump(info, open(results_file, 'wb'))

        if step_info['memory'] > 50 * 10**9:
            print('Memory limit reached! Stopping adaptive loop.')
            break


if __name__ == "__main__":
    run_adaptive_loop(rhs_functional_factory=example_rhs_functional,
                      u0_data=example_u0_data(),
                      initial_triangulation='unit_square',
                      results_file='smooth_solution_adaptive.pkl')
