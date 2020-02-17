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
from .error_estimator import AuxiliaryErrorEstimator, TimeSliceErrorEstimator
from .heat_equation import HeatEquation
from .heat_equation_test import (example_rhs_functional,
                                 example_solution_function, example_u0_data)


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


def test_heat_error_reduction():
    """ Simple test that applies the adaptive loop and checks slice errors. """
    # Adaptive parameters.
    theta = 0.3
    max_iters = 8
    solver_tol = 1e-3

    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

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

    # Create TimeSliceErrorEstimator.
    slice_error_estimator = TimeSliceErrorEstimator(
        *example_solution_function())

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

        # Append the residual norm, and the *real* number of dofs.
        res_errors.append(mark_info['res_norm'])
        dims.append(solve_info['dim_X_delta'])
        n_dofs.append(
            sum(1 for nv in u_dd_d.bfs()
                if not nv.nodes[1].on_domain_boundary))

        # Calculate the time slice errors
        cur_slice_errors = slice_error_estimator.estimate(u_dd_d, times)
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

            # Assert that we have a convergence rate of at least 0.4 :-).
            assert all(rates_slices > 0.4)
        if it > 7:
            # If we are futher enough, assert at least convergence of 0.5!
            assert all(rates_slices > 0.5)


def singular_u0_unit_square_data():
    return (lambda xy: 1, 1, 1.0)


def singular_u0_lshape_data():
    return (lambda xy: 1, 1, np.sqrt(3.0))


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


def mildly_singular_u0_data_unit():
    return (lambda xy: xy[0] * (xy[0] - 1) * xy[1] * (xy[1] - 1), 4, 1 / 30)


def mildly_singular_u0_data_lshape():
    return (lambda xy: xy[0] * (xy[0] - 1) * (xy[0] + 1) * xy[1] *
            (xy[1] - 1) * (xy[1] + 1), 6, 4 / 21 * np.sqrt(2 / 5))


def mildly_singular_rhs_functional_unit(heat_eq):
    g = [(
        lambda t: 1,
        lambda xy: 1,
    )]
    g_order = [(1, 1)]
    u0, u0_order, _ = mildly_singular_u0_data_unit()
    u0 = [u0]
    u0_order = [u0_order]

    return heat_eq.calculate_rhs_functionals_quadrature(g=g,
                                                        g_order=g_order,
                                                        u0=u0,
                                                        u0_order=u0_order)


def time_singular_solution_function(alpha=0.5):
    assert 0 < alpha <= 1
    u = (
        lambda t: 1 + t**alpha,
        lambda xy: (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1],
    )
    u_order = (5, 4)
    u_slice_norm_l2 = lambda t: (1 + t**alpha) / 30
    return u, u_order, u_slice_norm_l2


def time_singular_u0_data_unit(alpha=0.5):
    u, u_order, u_slice_norm_l2 = time_singular_solution_function(alpha)
    return (lambda xy: u[0](0) * u[1](xy), u_order[1], u_slice_norm_l2(0))


def time_singular_rhs_functional_unit(heat_eq, alpha=0.5):
    assert 0 < alpha <= 1
    g = [
        (
            lambda t: 2 * (1 + t**alpha),
            lambda xy: xy[0] * (1 - xy[0]) + xy[1] * (1 - xy[1]),
        ),
        (
            lambda t: alpha * t**(alpha - 1),
            lambda xy: xy[0] * (1 - xy[0]) * xy[1] * (1 - xy[1]),
        ),
    ]
    g_order = [(5, 2), (5, 4)]
    u0, u0_order, _ = time_singular_u0_data_unit()
    u0 = [u0]
    u0_order = [u0_order]

    return heat_eq.calculate_rhs_functionals_quadrature(g=g,
                                                        g_order=g_order,
                                                        u0=u0,
                                                        u0_order=u0_order)


def mildly_singular_rhs_functional_lshape(heat_eq):
    g = [(
        lambda t: 1,
        lambda xy: 1,
    )]
    g_order = [(1, 1)]
    u0, u0_order, _ = mildly_singular_u0_data_lshape()
    u0 = [u0]
    u0_order = [u0_order]

    return heat_eq.calculate_rhs_functionals_quadrature(g=g,
                                                        g_order=g_order,
                                                        u0=u0,
                                                        u0_order=u0_order)


def run_adaptive_loop(initial_triangulation='square',
                      theta=0.7,
                      results_file=None,
                      initial_refinement=1,
                      saturation_layers=1,
                      rhs_functional_factory=singular_rhs_functional,
                      u0_data=singular_u0_unit_square_data,
                      u_solution=None,
                      mean_zero=True,
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
    g_functional, u0_functional = rhs_functional_factory(HeatEquation(X_delta))

    # Create auxiliary error estimator.
    aux_error_estimator = AuxiliaryErrorEstimator(g_functional, u0_functional,
                                                  *u0_data)

    # Create time slice error estimator, if we have the real solution!
    if u_solution:
        slice_error_estimator = TimeSliceErrorEstimator(*u_solution)

    # Create adaptive heat equation object.
    adaptive_heat_eq = AdaptiveHeatEquation(
        X_init=X_delta,
        g_functional=g_functional,
        u0_functional=u0_functional,
        theta=theta,
        saturation_layers=saturation_layers)
    info = {
        'theta': adaptive_heat_eq.theta,
        'initial_refinement': initial_refinement,
        'saturation_layers': saturation_layers,
        'mean_zero': mean_zero,
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
            'u_delta':
            u_dd_d.to_array(),
        }
        assert len(sol_info['X_delta']) == len(sol_info['u_delta'])

        # Mark and refine.
        residual, mark_info = adaptive_heat_eq.mark_refine(u_dd_d=u_dd_d,
                                                           mean_zero=mean_zero)
        # Store residual information.
        step_info.update(mark_info)

        # Store the auxilary error estimator.
        aux_error, aux_terms = aux_error_estimator.estimate(
            adaptive_heat_eq.heat_dd_d, u_dd_d)
        step_info['aux_error'] = aux_error
        step_info['aux_terms'] = aux_terms

        # If we have the solution, also calculate some time slice errors.
        if u_solution:
            times = np.linspace(0, 1, 9)
            step_info['slice_times'] = times
            step_info['slice_errors'] = slice_error_estimator.estimate(
                u_dd_d, times)

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
    case = 'time'
    if case == 'smooth':
        run_adaptive_loop(
            rhs_functional_factory=example_rhs_functional,
            u0_data=example_u0_data(),
            u_solution=example_solution_function(),
            initial_triangulation='unit_square',
            saturation_layers=3,
            mean_zero=True,
            results_file='smooth_solution_adaptive_3layer_single.pkl')
    elif case == 'time':
        run_adaptive_loop(
            rhs_functional_factory=lambda heat_eq: time_singular_rhs_functional_unit(heat_eq,
            alpha=0.1),
            u_solution=time_singular_solution_function(alpha=0.1),
            u0_data=time_singular_u0_data_unit(alpha=0.1),
            initial_triangulation='unit_square',
            saturation_layers=1,
            mean_zero=True,
            results_file='time_solution_adaptive_unit_alpha0.1.pkl')
    elif case == 'mild':
        run_adaptive_loop(
            rhs_functional_factory=mildly_singular_rhs_functional_unit,
            u0_data=mildly_singular_u0_data_unit(),
            initial_triangulation='unit_square',
            saturation_layers=1,
            mean_zero=True,
            results_file='mild_solution_adaptive_unit.pkl')
    elif case == 'singular':
        run_adaptive_loop(
            rhs_functional_factory=singular_rhs_functional,
            u0_data=singular_u0_unit_square_data(),
            initial_triangulation='unit_square',
            saturation_layers=2,
            mean_zero=True,
            results_file='singular_solution_adaptive_2layer_single.pkl')
