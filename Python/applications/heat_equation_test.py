import os
import random
import time

import numpy as np
import pytest

from ..datastructures.applicator import LinearOperatorApplicator
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.tree_vector import TreeVector
from ..linalg.lanczos import Lanczos
from ..space.basis import HierarchicalBasisFunction
from ..space.operators import Operator
from ..space.triangulation import (InitialTriangulation,
                                   to_matplotlib_triangulation)
from ..space.triangulation_function import TriangulationFunction
from ..space.triangulation_view import TriangulationView
from ..spacetime.basis import generate_x_delta_underscore, generate_y_delta
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation
from .residual_error_estimator import ResidualErrorEstimator


def example_solution_function():
    u = (
        lambda t: 1 + t**2,
        lambda xy: (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1],
    )
    u_order = (2, 4)
    u_slice_norm_l2 = lambda t: (1 + t**2) / 30
    return u, u_order, u_slice_norm_l2


def example_rhs(heat_eq):
    g = [(
        lambda t: -2 * (1 + t**2),
        lambda xy: (xy[0] - 1) * xy[0] + (xy[1] - 1) * xy[1],
    ), (
        lambda t: 2 * t,
        lambda xy: (xy[0] - 1) * xy[0] * (xy[1] - 1) * xy[1],
    )]
    g_order = [(2, 2), (1, 4)]
    u, u_order, _ = example_solution_function()
    u0 = [lambda xy: u[0](0) * u[1](xy)]
    u0_order = [u_order[1]]

    return heat_eq.calculate_rhs_vector(
        *heat_eq.calculate_rhs_functionals_quadrature(
            g=g, g_order=g_order, u0=u0, u0_order=u0_order))


def random_rhs(heat_eq):
    # Create a (fake) tree for the rhs (X and Y) having random data.
    def call_random_fill(new_node, _):
        new_node.value = random.random()

    return heat_eq.create_vector(call_postprocess=call_random_fill)


def plot_slice(heat_eq, t, sol):
    """ Plots a slice of the given solution for a fixed time. """
    result = TreeVector.from_metaroot(sol.root.nodes[1])
    for nv in sol.project(0).bfs():
        # Check if t is contained inside support of time wavelet.
        a = float(nv.node.support[0].interval[0])
        b = float(nv.node.support[-1].interval[1])
        if a <= t <= b:
            result.axpy(nv.frozen_other_axis(), nv.node.eval(t))

    # Calculate the triangulation that is associated to the result.
    triang = TriangulationView(result)

    # Convert the result to single scale.
    space_operator = Operator(triang, heat_eq.dirichlet_boundary)
    result_ss = space_operator.apply_T(result.to_array())

    # Plot the result
    import matplotlib.pyplot as plt
    matplotlib_triang = to_matplotlib_triangulation(triang.elem_tree_view,
                                                    result)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(matplotlib_triang, Z=result_ss)
    plt.show()


def test_full_tensor_heat():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(4)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(4)

    # Create X^\delta
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(4)

    # Create heat equation object.
    for formulation in ['saddle', 'schur']:
        heat_eq = HeatEquation(X_delta=X_delta, formulation=formulation)
        rhs = random_rhs(heat_eq)

        # Try and apply the heat_eq block matrix to this rhs.
        tree_matvec = heat_eq.mat.apply(rhs)

        # Now do the same trick, but using vectors.
        array_matvec = heat_eq.linop.matvec(rhs.to_array())
        assert np.allclose(tree_matvec.to_array(), array_matvec)


def test_sparse_tensor_heat():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Create X^\delta
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.sparse_refine(2)

    # Create heat equation object.
    for formulation in ['saddle', 'schur']:
        heat_eq = HeatEquation(X_delta=X_delta, formulation=formulation)
        rhs = random_rhs(heat_eq)

        # Try and apply the heat_eq block matrix to this rhs.
        tree_matvec = heat_eq.mat.apply(rhs)

        # Now do the same trick, but using vectors.
        array_matvec = heat_eq.linop.matvec(rhs.to_array())
        assert np.allclose(tree_matvec.to_array(), array_matvec)

        # Now actually solve this beast!
        sol, info = heat_eq.solve(rhs)
        # Check the error..
        res_tree = heat_eq.mat.apply(sol)
        res_tree -= rhs
        assert np.linalg.norm(res_tree.to_array()) < 1e-4


def test_real_tensor_heat():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Create X^\delta
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.sparse_refine(3)

    # Create heat equation object.
    for formulation in ['saddle', 'schur']:
        heat_eq = HeatEquation(X_delta=X_delta, formulation=formulation)
        rhs = example_rhs(heat_eq)

        # Now actually solve this beast!
        sol, info = heat_eq.solve(rhs)
        error = np.linalg.norm(
            heat_eq.mat.apply(sol).to_array() - rhs.to_array())
        print('%s solved in {} iterations with an error {}'.format(
            formulation, info['num_iters'], error))

        # assert that solver converged.
        assert error < 1e-4

        # assert that the solution is not identically zero.
        u_sol = sol[1] if formulation == 'saddle' else sol
        assert sum(abs(sol.to_array())) > 0

    # Return heat_eq, sol for plotting purposes!
    return heat_eq, u_sol


def test_heat_eq_linear():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Create X^\delta
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.sparse_refine(2)

    # Create heat equation object.
    for formulation in ['saddle', 'schur']:
        heat_eq = HeatEquation(X_delta=X_delta, formulation=formulation)
        heat_eq_mat = heat_eq.linop.to_matrix()

        # Check that the heat_eq linear operator is linear.
        for _ in range(10):
            v = random_rhs(heat_eq)
            w = random_rhs(heat_eq)

            v_arr = v.to_array()
            w_arr = w.to_array()
            alpha = random.random()

            # Check whether the linop is linear.
            assert np.allclose(
                heat_eq.linop.matvec(v_arr + alpha * w_arr),
                heat_eq.linop.matvec(v_arr) +
                alpha * heat_eq.linop.matvec(w_arr))

            # Check whether the output corresponds to the matrix.
            assert np.allclose(heat_eq.linop.matvec(v_arr),
                               heat_eq_mat.dot(v_arr))


def test_heat_error_reduction(max_history_level=0,
                              max_level=6,
                              save_results_file=None,
                              formulation='saddle',
                              solver='minres'):
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(max_level)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(max_level)

    n_t = 9
    errors_quad = []
    ndofs = []
    dims = []
    time_per_dof = []
    rates_quad = []
    residual_norm_histories = []
    residual_norms = []
    solver_iters = []
    for level in range(2, max_level):
        # Create X^\delta as a sparse grid.
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.sparse_refine(level, weights=[2, 1])
        print('X_delta: dofs time axis={}\tdofs space axis={}'.format(
            len(X_delta.project(0).bfs()), len(X_delta.project(1).bfs())))

        # Create heat equation object.
        heat_eq = HeatEquation(X_delta=X_delta, formulation=formulation)
        rhs = example_rhs(heat_eq)

        if level <= max_history_level:
            residual_norm_history = []
            callback = lambda vec: residual_norm_history.append(
                np.linalg.norm(heat_eq.linop @ vec - rhs.to_array()))
        else:
            callback = None

        # Now actually solve this beast!
        sol, info = heat_eq.solve(rhs, solver=solver, iter_callback=callback)

        # Count number of dofs (not on the boundary!)
        ndofs.append(
            len([
                n for n in X_delta.bfs() if not n.nodes[1].on_domain_boundary
            ]))

        # Record some stuff for posterity.
        dims.append([len(heat_eq.Y_delta.bfs()), len(X_delta.bfs())])
        if level <= max_history_level:
            residual_norm_histories.append(residual_norm_history)
        solver_iters.append(info['num_iters'])
        residual_norm = np.linalg.norm(
            heat_eq.mat.apply(sol).to_array() - rhs.to_array())
        residual_norms.append(residual_norm)
        time_per_dof.append(info['time_per_dof'])

        print('{} solved in {} iterations with a residual norm {}'.format(
            solver, info['num_iters'], residual_norm))
        print('Time per dof is approximately {}'.format(info['time_per_dof']))

        u, u_order, u_slice_norm = example_solution_function()

        cur_errors_quad = np.ones(n_t)
        u_sol = sol[1] if formulation == 'saddle' else sol
        for i, t in enumerate(np.linspace(0, 1, n_t)):
            sol_slice = u_sol.slice(i=0,
                                    coord=t,
                                    slice_cls=TriangulationFunction)
            cur_errors_quad[i] = sol_slice.error_L2(
                lambda xy: u[0](t) * u[1](xy),
                u_slice_norm(t),
                u_order[1],
            )

        errors_quad.append(cur_errors_quad)
        if len(ndofs) == 1:
            rates_quad.append([0] * n_t)
        else:
            rates_quad.append(
                np.log(errors_quad[-1] / errors_quad[0]) /
                np.log(ndofs[0] / ndofs[-1]))

        print('-- Results for level = {} --'.format(level))
        print('\tdofs:', ndofs[-1])
        print('\ttime_per_dof: {0:.4f}'.format(time_per_dof[-1]))
        print('\terrors:', errors_quad[-1])
        print('\trates:', rates_quad[-1])
        print('\n')

        if save_results_file:
            import pickle
            results = {
                "n_t": n_t,
                "max_level": max_level,
                "dofs": ndofs,
                "dims": dims,
                "time_per_dof": time_per_dof,
                "residual_norm_histories": residual_norm_histories,
                "residual_norms": residual_norms,
                "solver_iters": solver_iters,
                "errors": errors_quad,
                "rates": rates_quad
            }
            pickle.dump(results, open(save_results_file, "wb"))

        if len(errors_quad) > 1:
            # Assert that at least 50% of the time steps have error reduction.
            assert sum(errors_quad[-1] <= errors_quad[-2]) > 0.5 * n_t

        if len(errors_quad) > 2:
            # Assert that at least 80% of the time steps have error reduction.
            assert sum(errors_quad[-1] <= errors_quad[-3]) > 0.8 * n_t

    # Assert that all our errors have reduced.
    assert all(errors_quad[-1] <= errors_quad[0])

    # Assert that we have a convergence rate of at least 0.25 :-).
    assert all(rates_quad[-1] > 0.25)

    # We expect a reat of atleast 0.5, but this requires some refines.
    if max_level >= 8: assert all(rates_quad[-1] > 0.5)


def test_preconditioned_eigenvalues(max_level=6, sparse_grid=True):
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(max_level)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(max_level)

    for level in range(2, max_level):
        # Create X^\delta as a sparse grid.
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        if sparse_grid:
            X_delta.sparse_refine(level, weights=[2, 1])
        else:
            X_delta.uniform_refine([level, 2 * level])
        print('X_delta: dofs time axis={}\tdofs space axis={}'.format(
            len(X_delta.project(0).bfs()), len(X_delta.project(1).bfs())))

        # Create heat equation object.
        heat_eq = HeatEquation(X_delta=X_delta, formulation='schur')
        S = heat_eq.linop
        Sinv = LinearOperatorApplicator(applicator=heat_eq.P_X,
                                        input_vec=heat_eq.create_vector())
        l = Lanczos(S, Sinv)
        assert l.cond() < 10
        print("Level {} with {} DoFs; l_min = {}; l_max = {}; kappa_2 = {}".
              format(level, len(X_delta.bfs()), l.lmin, l.lmax, l.cond()))


@pytest.mark.slow
def test_residual_error_estimator_rate():
    import psutil
    # Create space part.
    triang = InitialTriangulation.unit_square(initial_refinement=1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()

    max_level = 100
    rhs_factory = example_rhs
    sol = None
    time_start = time.time()
    for level in range(1, max_level):
        time_start_iteration = time.time()
        # Create X^\delta as a full grid.
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.uniform_refine(level)
        print([(n.nodes[0].level, n.nodes[1].level) for n in X_delta.bfs()
               if n.is_leaf()])
        X_dd, I_d_dd = generate_x_delta_underscore(X_delta)
        Y_dd = generate_y_delta(X_dd)
        heat_eq = HeatEquation(X_delta=X_delta,
                               Y_delta=Y_dd,
                               formulation='schur')
        rhs = rhs_factory(heat_eq)
        if sol:
            sol = sol.deep_copy()
            sol.union(X_delta, call_postprocess=None)
        sol, solve_info = heat_eq.solve(b=rhs, solver='pcg', x0=sol)
        error_estimator = ResidualErrorEstimator.FromDoubleTrees(
            u_dd_d=sol,
            rhs_factory=rhs_factory,
            X_d=X_delta,
            X_dd=X_dd,
            Y_dd=Y_dd,
            I_d_dd=I_d_dd)
        process = psutil.Process(os.getpid())
        print(len(X_delta.bfs()), len(X_dd.bfs()),
              process.memory_info().rss,
              time.time() - time_start_iteration,
              time.time() - time_start, error_estimator.res_dd_d.norm())
        if process.memory_info().rss > 40 * 10**9:
            break


if __name__ == "__main__":
    test_residual_error_estimator_rate()
    # test_preconditioned_eigenvalues(max_level=16, sparse_grid=True)
    test_heat_error_reduction(max_history_level=16,
                              max_level=16,
                              save_results_file=None,
                              formulation='schur',
                              solver='pcg')
