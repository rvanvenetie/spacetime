from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.tree_view import MetaRootView
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.haar_basis import HaarBasis


def test_double_tree_vector():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create a hierarchical basis
    hierarch_basis = MetaRootView(metaroot=triang.vertex_meta_root,
                                  node_view_cls=HierarchicalBasisFunction)
    hierarch_basis.deep_refine()

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create a DoubleTree Vector
    dt_root = DoubleTreeVector.full_tensor(
        HaarBasis.metaroot_wavelet,
        hierarch_basis,
        dbl_node_cls=DoubleNodeVector,
        frozen_dbl_cls=FrozenDoubleNodeVector)

    # Assert that main axis correspond to trees we've put in.
    assert [f_node.node for f_node in dt_root.project(0).bfs()
            ] == HaarBasis.metaroot_wavelet.bfs()
    assert [f_node.node
            for f_node in dt_root.project(1).bfs()] == hierarch_basis.bfs()

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
