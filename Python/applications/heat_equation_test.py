import cProfile
import random

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
from ..space.triangulation import InitialTriangulation
from ..spacetime.applicator import Applicator
from ..spacetime.basis import generate_y_delta
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation


def example_rhs(heat_eq):
    g = [(lambda t: 2 * t, \
          lambda x: x[0] * x[1] * (x[0]**2 - 1) * (x[1]**2 - 1)),
         (lambda t: -6 * (t**2 + 1), \
          lambda x: x[0] * x[1] * (x[0]**2 + x[1]**2 - 2))]
    g_order = (2, 6)

    u0 = lambda x: (x[0] - 1) * x[0] * (x[0] + 1) * (x[1] - 1) * x[1] * (x[1] +
                                                                         1)
    #    u0 = lambda x: (x[0] - 1) * x[0] * (x[1] - 1) * x[1]

    u0_order = 6

    result = heat_eq.calculate_rhs_vector(g=g,
                                          g_order=g_order,
                                          u0=u0,
                                          u0_order=u0_order)
    # Check that the vector != 0.
    assert sum(abs(result.to_array())) > 0.0001
    #    print('rhs vector g:', result[0].to_array())
    #    print('rhs vector -gamma u:', result[1].to_array())
    #    print(sum(abs(result.to_array())))
    return result


def random_rhs(heat_eq):
    # Create a (fake) tree for the rhs (X and Y) having random data.
    def call_random_fill(new_node, _):
        new_node.value = random.random()

    return heat_eq.create_vector(call_postprocess=call_random_fill)


def test_full_tensor_heat():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(5)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(5)

    # Create X^\delta
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.deep_refine()

    # Create heat equation obkect
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
    X_delta.sparse_refine(3)

    # Create heat equation obkect
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
    assert np.linalg.norm(res_tree.to_array()) < 1e-5


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

    # Create heat equation obkect
    heat_eq = HeatEquation(X_delta=X_delta)
    rhs = example_rhs(heat_eq)

    # Now actually solve this beast!
    sol, num_iters = heat_eq.solve(rhs)

    # Check the error..
    res_tree = heat_eq.mat.apply(sol)
    res_tree -= rhs
    assert np.linalg.norm(res_tree.to_array()) < 1e-5

    # Get the FEM solution for t = 0.5
    sol = res_tree[1]
    t = 0.5
    result = TreeVector(basis_space.root)
    for nv in sol.project(0).bfs():
        # Check if t is contained inside support of time wavelet.
        a = float(nv.node.support[0].interval[0])
        b = float(nv.node.support[-1].interval[1])
        if a <= t <= b:
            result.axpy(nv.frozen_other_axis(), nv.node.eval(t))

    print(result.bfs())


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

    # Create heat equation obkect
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


if __name__ == "__main__":
    cProfile.run('test_sparse_tensor_heat()', sort='tottime')
