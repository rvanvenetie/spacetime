from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.multi_tree_vector import BlockTreeVector
from ..datastructures.tree_view import TreeView
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


def test_double_tree_block_vector():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(5)

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(3)

    # Create two DoubleTree Vectors
    vec_1 = DoubleTreeVector(
        (HaarBasis.metaroot_wavelet, triang.vertex_meta_root))
    vec_2 = DoubleTreeVector(
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
