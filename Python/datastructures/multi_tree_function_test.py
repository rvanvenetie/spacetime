import numpy as np

from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.double_tree_function import DoubleTreeFunction
from ..space.triangulation import InitialTriangulation
from ..space.basis import HierarchicalBasisFunction
from ..time.three_point_basis import ThreePointBasis


def test_single_nonzero():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part.
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(1)

    tree = DoubleTree((basis_time.metaroot_wavelet, basis_space.root))
    tree.deep_refine()
    t, x, y = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j].reshape(3, -1)
    xy = np.vstack([x, y])

    interpolant = tree.deep_copy(
        mlt_node_cls=DoubleNodeVector,
        mlt_tree_cls=DoubleTreeFunction,
        call_postprocess=lambda nv, _: setattr(nv, 'value', 0.0))
    interpolant.bfs()[0].value = 1.0
    eval_interp = interpolant.eval(t, xy)
    print("hier")
    print(eval_interp.reshape(3, 3, 3))
