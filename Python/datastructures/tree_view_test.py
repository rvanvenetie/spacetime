from collections import deque

from ..space.triangulation import InitialTriangulation
from .tree_test import corner_index_tree, uniform_index_tree
from .tree_view import MetaRootView, NodeView


class FakeNodeView(NodeView):
    @property
    def level(self):
        return self.node.level


def test_deep_copy():
    # Generate some metaroots to work with.
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        metaroot_view = MetaRootView.from_metaroot(metaroot, FakeNodeView)
        metaroot_view.uniform_refine(max_level=10**9)
        assert [n.node for n in metaroot_view.bfs()] == metaroot.bfs()

        metaroot_copy = metaroot_view.deep_copy()
        assert [n.node for n in metaroot_copy.bfs()] == metaroot.bfs()
