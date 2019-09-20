from collections import deque

from .tree import MetaRoot
from .tree_view import NodeView, NodeVector
from ..space.triangulation import Triangulation


class FakeNodeView(NodeView):
    @property
    def level(self):
        if self.parents:
            return self.parents[0].level + 1
        else:
            return 0


def test_deep_copy():
    # Generate some metaroots to work with.
    T = Triangulation.unit_square()
    for _ in range(4):
        T.refine_uniform()

    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        metaroot_view = MetaRoot(
            [FakeNodeView(root) for root in metaroot.roots])
        metaroot_view.uniform_refine(max_level=4)
        print(set([n.node for n in metaroot_view.bfs()]) ^ set(metaroot.bfs()))
        #TreePlotter.draw_matplotlib_graph(metaroot)
        #TreePlotter.draw_matplotlib_graph(metaroot_view)
        #plt.show()
        assert [n.node for n in metaroot_view.bfs()] == metaroot.bfs()

        metaroot_copy = MetaRoot(
            [root.deep_copy() for root in metaroot_view.roots])
        assert [n.node for n in metaroot_copy.bfs()] == metaroot.bfs()


def test_add():
    T = Triangulation.unit_square()
    T.refine_uniform()
