import random
from collections import defaultdict

import numpy as np

from ..datastructures.multi_tree_function import (DoubleTreeFunction,
                                                  TreeFunction)
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..space.triangulation_function import TriangulationFunction
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


def test_slice():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.vertex_meta_root.uniform_refine(4)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(4)

    # Create random (uniform) vector.
    vec = DoubleTreeFunction.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    vec.uniform_refine(
        4, call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    vec_sliced = vec.slice(i=0, coord=0,
                           slice_cls=TriangulationFunction).to_array()

    # True value.
    vec_true = np.zeros(len(vec_sliced))
    for nv in vec.project(0).bfs():
        if nv.node.level == 0 or nv.node.index == 0:
            vec_true += nv.node.eval(0) * nv.frozen_other_axis().to_array()
    assert np.allclose(vec_true, vec_sliced)

    # Create random sparse vector.
    vec = DoubleTreeFunction.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    vec.sparse_refine(
        2, [2, 1],
        call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    vec_sliced = vec.slice(i=0, coord=0, slice_cls=TriangulationFunction)

    # True value.
    vec_true = defaultdict(float)
    for nv in vec.project(0).bfs():
        if nv.node.level == 0 or nv.node.index == 0:
            val = nv.node.eval(0)
            for phi in nv.frozen_other_axis().bfs():
                vec_true[phi.node] += val * phi.value

    for nv in vec_sliced.bfs():
        assert np.allclose(nv.value, vec_true[nv.node])


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
