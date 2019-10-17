import random
from collections import defaultdict

import numpy as np

from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..datastructures.tree_view import TreeView
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.haar_basis import HaarBasis


def test_double_tree_vector():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create a DoubleTree Vector
    dt_root = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    dt_root.uniform_refine()

    # Assert that main axis correspond to trees we've put in.
    assert [f_node.node for f_node in dt_root.project(0).bfs()
            ] == HaarBasis.metaroot_wavelet.bfs()
    assert [f_node.node for f_node in dt_root.project(1).bfs()
            ] == triang.vertex_meta_root.bfs()

    # Initialize the vector ones.
    for db_node in dt_root.bfs():
        db_node.value = 1
    for db_node in dt_root.bfs():
        assert db_node.value == 1

    # Check that this also holds for the fibers.
    for labda in dt_root.project(0).bfs():
        fiber = dt_root.fiber(1, labda)
        assert all(f_node.value == 1 for f_node in fiber.bfs())

    for labda in dt_root.project(1).bfs():
        fiber = dt_root.fiber(0, labda)
        assert all(f_node.value == 1 for f_node in fiber.bfs())

    # Check that the to_array is correct.
    np_vec = dt_root.to_array()
    assert len(np_vec) == len(dt_root.bfs())
    assert all(val == 1 for val in np_vec)

    # Check that from_array also works.
    dt_np = dt_root.deep_copy()
    dt_np.from_array(np_vec)
    assert len(dt_np.bfs()) == len(dt_root.bfs())
    for db_node in dt_np.bfs():
        assert db_node.value == 1

    # Check that copying works.
    dt_copy = dt_root.deep_copy()
    assert len(dt_copy.bfs()) == len(dt_root.bfs())
    for db_node in dt_copy.bfs():
        assert db_node.value == 1


def test_double_tree_vector_sum():
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(6)
    HaarBasis.metaroot_wavelet.uniform_refine(4)

    # Create a sparse DoubleTree Vector filled with random values.
    vec_sp = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    vec_sp.sparse_refine(
        4, call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    # Create a double tree uniformly refined with levels [1,4].
    vec_unif = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    vec_unif.uniform_refine(
        [1, 4],
        call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    # Create two empty vectors holding the sum.
    vec_0 = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    vec_1 = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))

    # vec_0 = vec_sp + vec_unif
    vec_0 += vec_sp
    vec_0 += vec_unif

    # vec_1 = vec_unif + vec_sp
    vec_1 += vec_unif
    vec_1 += vec_sp

    # Assert vec_0 == vec_1
    assert [(nv.nodes, nv.value) for nv in vec_0.bfs()
            ] == [(nv.nodes, nv.value) for nv in vec_1.bfs()]

    # Assert that the sum domain is larger than the two vectors themselves.
    assert len(vec_0.bfs()) > max(len(vec_sp.bfs()), len(vec_unif.bfs()))

    # Calculate the sum by dict
    vec_dict = defaultdict(float)
    for nv in vec_sp.bfs():
        vec_dict[tuple(nv.nodes)] = nv.value
    for nv in vec_unif.bfs():
        vec_dict[tuple(nv.nodes)] += nv.value

    for nv in vec_0.bfs():
        assert nv.value == vec_dict[tuple(nv.nodes)]


def test_initialize_quadrature():
    # g is the RHS to u(t,x,y) = (t**2 + 1) (x-1) x (x+1) (y-1) y (y+1).
    # so g(t,x,y) = 2xy(t(x^2-1)(y^2-1) - 3(t^2 + 1)(x^2 + y^2 - 2))
    #             = 2t * xy(x^2-1)(y^2-1) - 6(t^2 + 1) * xy(x^2 + y^2 - 2).
    g = [(lambda t: 2 * t, \
          lambda x: x[0] * x[1] * (x[0]**2 - 1) * (x[1]**2 - 1)),
         (lambda t: -6 * (t**2 + 1), \
          lambda x: x[0] * x[1] * (x[0]**2 + x[1]**2 - 2))]
    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create a DoubleTree Vector
    dt_root = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    dt_root.uniform_refine()

    # Initialize the vector.
    for db_node in dt_root.bfs():
        hbf = HierarchicalBasisFunction(db_node.nodes[1])
        db_node.value = sum(db_node.nodes[0].inner_quad(g1, g_order=2) *
                            hbf.inner_quad(g2, g_order=6) for (g1, g2) in g)

    for db_node in dt_root.bfs():
        assert np.isfinite(db_node.value)


def test_double_tree_block_vector():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(5)

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(3)

    # Create two DoubleTree Vectors
    vec_1 = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    vec_2 = DoubleTreeVector.from_metaroots(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))

    # Refine accroding to different levels, and store different values.
    vec_1.uniform_refine([2, 3],
                         call_postprocess=lambda nv: setattr(nv, 'value', 1))
    vec_2.uniform_refine([4, 2],
                         call_postprocess=lambda nv: setattr(nv, 'value', 2))

    block_vec = BlockTreeVector((vec_1, vec_2))

    arr_block_vec = block_vec.to_array()
    assert sum(arr_block_vec) == len(vec_1.bfs()) + 2 * len(vec_2.bfs())

    # Minus everything.
    arr_block_vec = -arr_block_vec

    # Set back these values.
    block_vec.from_array(arr_block_vec)

    for nv in vec_1.bfs():
        assert nv.value == -1
    for nv in vec_2.bfs():
        assert nv.value == -2
