import cProfile
import random

from .. import space, time
from ..datastructures.applicator import BlockApplicator
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..space.triangulation import InitialTriangulation
from ..spacetime.applicator import Applicator
from ..spacetime.basis import generate_y_delta
from ..time.three_point_basis import ThreePointBasis
from .heat_equation import HeatEquation


def random_rhs(heat_eq):
    # Create a (fake) tree for the rhs (X and Y) having random data.
    def call_random_fill(_, new_node):
        new_node.value = random.random()

    rhs_y = heat_eq.Y_delta.deep_copy(mlt_node_cls=DoubleNodeVector,
                                      mlt_tree_cls=DoubleTreeVector,
                                      call_postprocess=call_random_fill)
    rhs_x = heat_eq.X_delta.deep_copy(mlt_node_cls=DoubleNodeVector,
                                      mlt_tree_cls=DoubleTreeVector,
                                      call_postprocess=call_random_fill)
    return (rhs_y, rhs_x)


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
    heat_eq.mat.apply(rhs)


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
    X_delta.sparse_refine(7)

    # Create heat equation obkect
    heat_eq = HeatEquation(X_delta=X_delta)
    rhs = random_rhs(heat_eq)

    # Try and apply the heat_eq block matrix to this rhs.
    heat_eq.mat.apply(rhs)


if __name__ == "__main__":
    cProfile.run('test_sparse_tensor_heat()', sort='tottime')
