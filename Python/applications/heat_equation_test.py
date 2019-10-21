import cProfile
import random
import pytest

import numpy as np

from .. import space, time
from ..datastructures.applicator import (BlockApplicator,
                                         LinearOperatorApplicator)
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..datastructures.tree_vector import TreeVector
from ..space.basis import HierarchicalBasisFunction
from ..space.operators import Operator
from ..space.triangulation import (InitialTriangulation,
                                   to_matplotlib_triangulation)
from ..space.triangulation_view import TriangulationView
from ..space.triangulation_function import TriangulationFunction
from ..spacetime.applicator import Applicator
from ..spacetime.basis import generate_y_delta
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation


def example_solution_function():
    u = (lambda t: 1 + t**2, lambda xy: (1 - xy[0]) * xy[0] *
         (1 - xy[1]) * xy[1])
    u_order = (2, 4)
    return u, u_order


def example_rhs(heat_eq):
    g = [(lambda t: -2 * (1 + t**2), lambda xy: (xy[0] - 1) * xy[0] +
          (xy[1] - 1) * xy[1]),
         (lambda t: 2 * t, lambda xy: (xy[0] - 1) * xy[0] *
          (xy[1] - 1) * xy[1])]
    g_order = (2, 4)
    u, u_order = example_solution_function()
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


@pytest.mark.slow
def test_heat_refine():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    n_t = 101
    prev_errors = np.ones(n_t)
    for level in range(2, 6):
        # Create X^\delta
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.sparse_refine(level)

        # Create heat equation object.
        heat_eq = HeatEquation(X_delta=X_delta)
        rhs = example_rhs(heat_eq)

        # Now actually solve this beast!
        sol, num_iters = heat_eq.solve(rhs)
        error = np.linalg.norm(
            heat_eq.mat.apply(sol).to_array() - rhs.to_array())
        print('MINRES solved in {} iterations with an error {}'.format(
            num_iters, error))
        u, order = example_solution_function()

        for i, t in enumerate(np.linspace(0, 1, n_t)):
            sol_slice = sol[1].slice_time(t, slice_cls=TriangulationFunction)
            true_slice = sol_slice.deep_copy().from_singlescale_array(
                np.array([
                    u[0](t) * u[1](node.node.node.xy)
                    for node in sol_slice.bfs()
                ]))
            sol_slice -= true_slice
            error = sol_slice.L2norm()
            assert error < prev_errors[i]
            prev_errors[i] = error


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    heat_eq, sol = test_real_tensor_heat()
    fig = plt.figure()
    for t in np.linspace(0, 1, 100):
        plt.clf()
        sol.slice_time(t).plot(fig=fig, show=False)
        plt.show(block=False)
        plt.pause(0.05)
    plt.show()
