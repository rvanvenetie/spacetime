import os
import random
import time
from pprint import pprint

import numpy as np
import psutil

from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..space.triangulation_function import TriangulationFunction
from ..time.three_point_basis import ThreePointBasis
from .adaptive_heat_equation import AdaptiveHeatEquation
from .heat_equation import HeatEquation
from .heat_equation_test import (example_rhs_functional,
                                 example_solution_function)


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


def test_heat_error_reduction(theta=0.7,
                              max_iters=3,
                              initial_refinement=1,
                              solver_tol=1e-5,
                              save_results_file=None):
    """ Simple test that applies the adaptive loop for a few iterations. """
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    triang = InitialTriangulation.unit_square(
        initial_refinement=initial_refinement)
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
    n_t = 9
    times = np.linspace(0, 1, n_t)
    u_dd_d = None
    n_dofs = []
    dims = []
    res_errors = []
    slice_errors = []
    rates_slices = None
    for it in range(max_iters):
        u_dd_d, solve_info = adaptive_heat_eq.solve_step(x0=u_dd_d,
                                                         solver_tol=solver_tol)
        residual, mark_info = adaptive_heat_eq.mark_refine(u_dd_d=u_dd_d)

        print('\n')
        pprint(solve_info)
        pprint(mark_info)

        # Append the residual norm, and the *real* number of dofs.
        res_errors.append(mark_info['res_norm'])
        dims.append(solve_info['dim_X_delta'])
        n_dofs.append(
            sum(1 for nv in u_dd_d.bfs()
                if not nv.nodes[1].on_domain_boundary))

        # Retrieve the exact solution and do calculate errors for time slices.
        u, u_order, u_slice_norm = example_solution_function()
        cur_slice_errors = np.ones(n_t)
        u_sol = u_dd_d
        for i, t in enumerate(times):
            sol_slice = u_sol.slice(i=0,
                                    coord=t,
                                    slice_cls=TriangulationFunction)
            cur_slice_errors[i] = sol_slice.error_L2(
                lambda xy: u[0](t) * u[1](xy),
                u_slice_norm(t),
                u_order[1],
            )
        slice_errors.append(cur_slice_errors)
        if it > 2:
            rates_slices = np.log(
                slice_errors[-1] / slice_errors[-3]) / np.log(
                    n_dofs[-3] / n_dofs[-1])
        print('\n')
        print('-- Results for iter = {} --'.format(it + 1))
        print('dofs:', n_dofs[-1])
        print('residual error: {:.5g}'.format(res_errors[-1]))
        if it > 2:
            print('slice\ttime\trate')
            for i, t in enumerate(times):
                print('{}\t{}\t{:.3f}'.format(
                    i, t, -1 if it == 0 else rates_slices[i]))
        print('\n')

        if save_results_file:
            import pickle
            results = {
                "times": times,
                "dofs": n_dofs,
                "theta": theta,
                "dims": dims,
                "residual_errors": res_errors,
                "slice_errors": slice_errors,
            }
            pickle.dump(results, open(save_results_file, "wb"))
        continue

        # Do some assertion checks.
        if it > 2:
            # Assert that at least 50% of the time steps have error reduction.
            assert sum(slice_errors[-1] <= slice_errors[-2]) > 0.5 * n_t

            # Assert that the residual norm has reduced.
            assert res_errors[-1] < 0.1

        if it > 4:
            # Assert that at least 80% of the time steps have error reduction.
            assert sum(slice_errors[-1] <= slice_errors[-3]) > 0.8 * n_t
        if it > 6:
            # Assert that the residual norm has reduced even futher.
            assert res_errors[-1] < 0.08

            # Assert that all our slice errors have reduced.
            assert all(slice_errors[-1] <= slice_errors[0])

            # Assert that we have a convergence rate of at least 0.25 :-).
            assert all(rates_slices > 0.25)
        if it > 8:
            # If we are futher enough, assert at least convergence of 0.5!
            assert all(rates_slices > 0.5)


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
                      rhs_factory=singular_rhs_functional,
                      solver_tol=1e-7):
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
    time_start = time.time()
    while True:
        time_start_iteration = time.time()
        step_info = {}
        # Calculate a new solution.
        u_dd_d, solve_info = adaptive_heat_eq.solve_step(x0=u_dd_d,
                                                         solver='pcg',
                                                         solver_tol=solver_tol)
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
        assert len(sol_info['X_delta']) == len(sol_info['u_delta'])

        # Mark and refine.
        residual, mark_info = adaptive_heat_eq.mark_refine(u_dd_d=u_dd_d)
        step_info.update(mark_info)
        sol_info['residual'] = residual.to_array()

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
    # run_adaptive_loop(rhs_factory=singular_rhs_functional,
    #                   initial_triangulation='lshape',
    #                   results_file='singular_solution_adaptive_lshape.pkl')
    test_heat_error_reduction(
        save_results_file='slice_errors_adaptive_initial_ref_5.pkl',
        solver_tol=1e-6,
        initial_refinement=5,
        max_iters=9999)
