import numpy as np

from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.tree_view import TreeView
from ..space.triangulation import InitialTriangulation
from ..space.basis import HierarchicalBasisFunction
from ..time.haar_basis import HaarBasis


def test_double_tree_vector():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create a hierarchical basis
    hierarch_basis = TreeView(triang.vertex_meta_root)
    hierarch_basis.deep_refine()

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create a DoubleTree Vector
    dt_root = DoubleTreeVector.from_metaroots(
        HaarBasis.metaroot_wavelet,
        triang.vertex_meta_root,
        dbl_node_cls=DoubleNodeVector,
        frozen_dbl_cls=FrozenDoubleNodeVector)
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
    def call_copy_value(new_node, old_node):
        new_node.value = old_node.value

    dt_copy = dt_root.deep_copy(call_postprocess=call_copy_value)
    assert len(dt_copy.bfs()) == len(dt_root.bfs())
    for db_node in dt_copy.bfs():
        assert db_node.value == 1


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
        HaarBasis.metaroot_wavelet,
        triang.vertex_meta_root,
        dbl_node_cls=DoubleNodeVector,
        frozen_dbl_cls=FrozenDoubleNodeVector)
    dt_root.uniform_refine()

    # Initialize the vector.
    for db_node in dt_root.bfs():
        db_node.value = sum(
            db_node.nodes[0].inner_quad(g1) *
            HierarchicalBasisFunction(db_node.nodes[1]).inner_quad(g2, order=6)
            for (g1, g2) in g)

    for db_node in dt_root.bfs():
        assert np.isfinite(db_node.value)
