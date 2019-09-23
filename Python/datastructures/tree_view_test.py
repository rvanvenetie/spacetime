from collections import deque

from .tree_test import uniform_index_tree, corner_index_tree
from .tree_view import NodeView, MetaRootView
from ..space.triangulation import Triangulation


class FakeNodeView(NodeView):
    @property
    def level(self):
        return self.node.labda[0]


def test_deep_copy():
    # Generate some metaroots to work with.
    T = Triangulation.unit_square()
    for _ in range(5):
        T.refine_uniform()
    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        metaroot_view = MetaRootView(
            [FakeNodeView(root) for root in metaroot.roots])
        metaroot_view.uniform_refine(max_level=10**9)
        assert [n.node for n in metaroot_view.bfs()] == metaroot.bfs()

        metaroot_copy = metaroot_view.deep_copy()
        assert [n.node for n in metaroot_copy.bfs()] == metaroot.bfs()
