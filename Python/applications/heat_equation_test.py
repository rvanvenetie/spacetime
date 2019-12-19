import random

import numpy as np

from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.tree_vector import TreeVector
from ..space.basis import HierarchicalBasisFunction
from ..space.operators import Operator
from ..space.triangulation import (InitialTriangulation,
                                   to_matplotlib_triangulation)
from ..space.triangulation_function import TriangulationFunction
from ..space.triangulation_view import TriangulationView
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation


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
    g_order = (2, 4)
    u, u_order, _ = example_solution_function()
    u0 = lambda xy: u[0](0) * u[1](xy)
    u0_order = u_order[1]

    result = heat_eq.calculate_rhs_vector(g=g,
                                          g_order=g_order,
                                          u0=u0,
                                          u0_order=u0_order)
    # Check that the vector != 0.
    assert sum(abs(result.to_array())) > 0.0001

    return result


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
    heat_eq = HeatEquation(X_delta=X_delta)
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
    heat_eq = HeatEquation(X_delta=X_delta)
    rhs = random_rhs(heat_eq)

    # Try and apply the heat_eq block matrix to this rhs.
    tree_matvec = heat_eq.mat.apply(rhs)

    # Now do the same trick, but using vectors.
    array_matvec = heat_eq.linop.matvec(rhs.to_array())
    assert np.allclose(tree_matvec.to_array(), array_matvec)

    # Now actually solve this beast!
    sol, num_iters = heat_eq.solve(rhs)

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
    heat_eq = HeatEquation(X_delta=X_delta)
    rhs = example_rhs(heat_eq)

    # Now actually solve this beast!
    sol, num_iters = heat_eq.solve(rhs)
    error = np.linalg.norm(heat_eq.mat.apply(sol).to_array() - rhs.to_array())
    print('MINRES solved in {} iterations with an error {}'.format(
        num_iters, error))

    # assert that minres converged.
    assert error < 1e-4

    # assert that the solution is not identically zero.
    assert sum(abs(sol[1].to_array())) > 0

    # Return heat_eq, sol for plotting purposes!
    return heat_eq, sol[1]


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
    heat_eq = HeatEquation(X_delta=X_delta)
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
            heat_eq.linop.matvec(v_arr) + alpha * heat_eq.linop.matvec(w_arr))

        # Check whether the output corresponds to the matrix.
        assert np.allclose(heat_eq.linop.matvec(v_arr), heat_eq_mat.dot(v_arr))


def test_heat_error_reduction(max_history_level=0,
                              max_level=6,
                              save_results_file=None):
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
    minres_iters = []
    for level in range(2, max_level):
        # Create X^\delta as a sparse grid.
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.sparse_refine(level, weights=[2, 1])
        print('X_delta: dofs time axis={}\tdofs space axis={}'.format(
            len(X_delta.project(0).bfs()), len(X_delta.project(1).bfs())))

        # Create heat equation object.
        heat_eq = HeatEquation(X_delta=X_delta)
        rhs = example_rhs(heat_eq)

        if level <= max_history_level:
            residual_norm_history = []
            callback = lambda vec: residual_norm_history.append(
                np.linalg.norm(heat_eq.linop @ vec - rhs.to_array()))
        else:
            callback = None
        # Now actually solve this beast!
        sol, num_iters = heat_eq.solve(rhs, iter_callback=callback)

        # Count number of dofs (not on the boundary!)
        ndofs.append(
            len([
                n for n in X_delta.bfs() if not n.nodes[1].on_domain_boundary
            ]))

        # Record some stuff for posterity.
        dims.append([len(heat_eq.Y_delta.bfs()), len(X_delta.bfs())])
        if level <= max_history_level:
            residual_norm_histories.append(residual_norm_history)
        minres_iters.append(num_iters)
        residual_norm = np.linalg.norm(
            heat_eq.mat.apply(sol).to_array() - rhs.to_array())
        residual_norms.append(residual_norm)
        time_per_dof.append(heat_eq.time_per_dof())

        print('MINRES solved in {} iterations with a residual norm {}'.format(
            num_iters, residual_norm))
        print('Time per dof is approximately {}'.format(
            heat_eq.time_per_dof()))

        u, u_order, u_slice_norm = example_solution_function()

        cur_errors_quad = np.ones(n_t)
        for i, t in enumerate(np.linspace(0, 1, n_t)):
            sol_slice = sol[1].slice(i=0,
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
                "minres_iters": minres_iters,
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


if __name__ == "__main__":
    test_heat_error_reduction(
        max_history_level=9,
        max_level=16,
        save_results_file='error_reduction.pickle',
    )
