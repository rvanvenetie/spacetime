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
from ..space.triangulation import InitialTriangulation
from ..spacetime.applicator import Applicator
from ..spacetime.basis import generate_y_delta
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation


def random_rhs(heat_eq):
    # Create a (fake) tree for the rhs (X and Y) having random data.
    def call_random_fill(new_node, _):
        new_node.value = random.random()

    return heat_eq.create_vector(call_postprocess=call_random_fill)


def test_full_tensor_heat():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    basis_space = triang.vertex_meta_root
    basis_space.uniform_refine(5)

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(5)

    # Create X^\delta
    X_delta = DoubleTree((basis_time.metaroot_wavelet, basis_space))
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
    basis_space = triang.vertex_meta_root
    basis_space.uniform_refine(6)

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Create X^\delta
    X_delta = DoubleTree((basis_time.metaroot_wavelet, basis_space))
    X_delta.sparse_refine(2)

    # Create heat equation obkect
    heat_eq = HeatEquation(X_delta=X_delta)
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
    assert np.linalg.norm(res_tree.to_array()) < 1e-5


def test_heat_eq_linear():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    basis_space = triang.vertex_meta_root
    basis_space.uniform_refine(6)

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Create X^\delta
    X_delta = DoubleTree((basis_time.metaroot_wavelet, basis_space))
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
