import numpy as np
import time

from ..datastructures.multi_tree_function import (DoubleTreeFunction,
                                                  TreeFunction)
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis


def test_dbl_fn_single_nonzero():
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
    dbl_fn.uniform_refine(1)
    t, x, y = np.mgrid[0:1:25j, 0:1:25j, 0:1:25j].reshape(3, -1)
    xy = np.vstack([x, y])

    for dbl_node in dbl_fn.bfs():
        dbl_node.value = 1.0
        assert np.allclose(
            dbl_fn.eval((t, xy)),
            dbl_node.nodes[0].eval(t) * dbl_node.nodes[1].eval(xy))
        dbl_node.value = 0.0


def test_fn_small():
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(1)
    fn = TreeFunction.from_metaroot(basis_time.metaroot_wavelet)
    fn.uniform_refine()
    t = np.linspace(0, 1, 1001)
    for node in fn.bfs():
        node.value = 1.0
        assert np.allclose(fn.eval(t), node.node.eval(t))
        node.value = 0.0


if __name__ == "__main__":
    """ We can plot a slice of a DoubleTreeFunction. Animate it for fun. """
    import matplotlib.pyplot as plt
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(2)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part.
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(2)

    dbl_fn = DoubleTreeFunction.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    dbl_fn.uniform_refine()
    dbl_fn.bfs()[7].value = 1.0
    dbl_fn.bfs()[2].value = 1.0
    fig = plt.figure()
    for t in np.linspace(0, 1, 100):
        plt.clf()
        dbl_fn.slice_time(t).plot(fig=fig, show=False)
        plt.show(block=False)
        plt.pause(0.05)
    plt.show()
