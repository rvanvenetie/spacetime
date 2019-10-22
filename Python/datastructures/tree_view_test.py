
from ..space.triangulation import InitialTriangulation
from .tree_view import TreeView


def test_deep_copy():
    # Generate some metaroots to work with.
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        metaroot_view = TreeView.from_metaroot(metaroot)
        metaroot_view.uniform_refine(max_level=10**9)
        assert [n.node for n in metaroot_view.bfs()] == metaroot.bfs()

        metaroot_copy = metaroot_view.deep_copy()
        assert [n.node for n in metaroot_copy.bfs()] == metaroot.bfs()


def test_uniform_refine():
    # Generate some metaroots to work with.
    T = InitialTriangulation.unit_square()
    metaroot_view = TreeView.from_metaroot(T.elem_meta_root)
    assert len(metaroot_view.bfs()) == 0
    metaroot_view.uniform_refine(5)
    assert len(metaroot_view.bfs()) == 2

    # Refine the underlying tree.
    T.elem_meta_root.uniform_refine(2)
    assert len(T.elem_meta_root.bfs()) == 2 + 2 * 2 + 2 * 2 * 2
    assert len(metaroot_view.bfs()) == 2
    metaroot_view.uniform_refine(1)
    assert len(metaroot_view.bfs()) == 2 + 2 * 2
    metaroot_view.uniform_refine(2)
    assert len(metaroot_view.bfs()) == 2 + 2 * 2 + 2 * 2 * 2
    metaroot_view.uniform_refine(10)
    assert len(metaroot_view.bfs()) == 2 + 2 * 2 + 2 * 2 * 2
