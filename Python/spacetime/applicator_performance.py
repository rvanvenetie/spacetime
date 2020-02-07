import numpy as np

from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.double_tree_view import DoubleTree
from ..space import applicator as s_applicator
from ..space import operators as s_operators
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..spacetime.basis import generate_y_delta
from ..time import applicator as t_applicator
from ..time import operators as t_operators
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .applicator import Applicator

seed = 0


def bsd_rnd():
    global seed
    seed = (1103515245 * seed + 12345) & 0x7fffffff
    return seed


def test_python(level, bilform_iters, inner_iters):
    use_cache = True
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(level)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time_X = ThreePointBasis()
    basis_time_X.metaroot_wavelet.uniform_refine(level)
    basis_time_Y = OrthonormalBasis()
    basis_time_Y.metaroot_wavelet.uniform_refine(level)

    for _ in range(bilform_iters):
        # Create X^\delta
        X_delta = DoubleTree.from_metaroots(
            (basis_time_X.metaroot_wavelet, basis_space.root))
        X_delta.deep_refine(lambda nv: nv[0].level <= 0 or nv[1].level <= 0 or
                            (bsd_rnd() % 3) != 0)
        Y_delta = generate_y_delta(X_delta)
        vec_X = X_delta.deep_copy(DoubleTreeVector)
        vec_Y = Y_delta.deep_copy(DoubleTreeVector)
        B = Applicator(Lambda_in=X_delta,
                       Lambda_out=Y_delta,
                       applicator_time=t_applicator.Applicator(
                           t_operators.transport(basis_time_X, basis_time_Y),
                           basis_in=basis_time_X,
                           basis_out=basis_time_Y),
                       applicator_space=s_applicator.Applicator(
                           s_operators.MassOperator(), use_cache=use_cache))
        for _ in range(inner_iters):
            for nv in vec_X.bfs():
                nv.value = bsd_rnd()
            B.apply(vec_in=vec_X, vec_out=vec_Y)


if __name__ == "__main__":
    test_python(level=5, bilform_iters=20, inner_iters=5)
