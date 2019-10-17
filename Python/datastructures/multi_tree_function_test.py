import numpy as np

from ..datastructures.multi_tree_function import (MultiTreeFunction,
                                                  DoubleTreeFunction)
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

    dbl_fn = DoubleTreeFunction.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    dbl_fn.uniform_refine()
    t, x, y = np.mgrid[0:1:25j, 0:1:25j, 0:1:25j].reshape(3, -1)
    xy = np.vstack([x, y])

    for dbl_node in dbl_fn.bfs():
        dbl_node.value = 1.0
        assert np.allclose(
            dbl_fn.eval((t, xy)),
            dbl_node.nodes[0].eval(t) * dbl_node.nodes[1].eval(xy))
        dbl_node.value = 0.0
