from collections import deque
import random
import numpy as np

from .tree_test import uniform_index_tree, corner_index_tree
from .tree_view import NodeView, MetaRootView, NodeVector
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


def test_vector_add():
    # Generate some metaroots to work with.
    T = Triangulation.unit_square()
    for _ in range(4):
        T.refine_uniform()
    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        vec = MetaRootView([NodeVector(root, 0.0) for root in metaroot.roots])
        vec.uniform_refine(max_level=5)
        for node in vec.bfs():
            node.value = random.random()

        # Make a copy, assert the values are close.
        vec_copy = vec.deep_copy()
        assert np.allclose([a.value for a in vec.bfs()],
                           [b.value for b in vec_copy.bfs()])

        # Add something to vec_copy, assert some things.
        vec_values = [a.value for a in vec.bfs()]
        vec_copy += vec
        assert np.allclose(vec_values, [a.value for a in vec.bfs()])
        assert np.allclose([2 * a.value for a in vec.bfs()],
                           [b.value for b in vec_copy.bfs()])

        # Multiply vec, assert some things.
        vec *= 1.5
        assert np.allclose([v * 1.5 for v in vec_values],
                           [a.value for a in vec.bfs()])

        # Create a unit vector on a coarser grid.
        vec2 = MetaRootView([NodeVector(root, 0.0) for root in metaroot.roots])
        vec2.uniform_refine(max_level=2)
        for node in vec2.bfs():
            node.value = 1.0
        assert len(vec2.bfs()) < len(vec.bfs())

        # Add vec to it, which is defined on a finer grid.
        vec2 += vec
        assert len(vec2.bfs()) == len(vec.bfs())
        # Assert that `vec2` now indeed has 1+v[node] for node.level <= 1,
        # and v[node] for node.level > 1.
        assert np.allclose(
            [b.value for b in vec2.bfs()],
            [a.value + (1 if a.level <= 2 else 0) for a in vec.bfs()])