import random

import numpy as np

from ..space.triangulation import InitialTriangulation
from .tree_test import corner_index_tree, uniform_index_tree
from .tree_vector import NodeVector, TreeVector
from .tree_view import NodeView, TreeView


def test_vector_add():
    # Generate some metaroots to work with.
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(4)
    for metaroot in [T.elem_meta_root, T.vertex_meta_root]:
        vec = TreeVector.from_metaroot(metaroot)
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
        vec2 = TreeVector.from_metaroot(metaroot)
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
