from collections import defaultdict

import numpy as np

from ..datastructures.tree_view import MetaRootView, NodeView
from .triangulation import InitialTriangulation, elem_tree_from_vertex_tree


def test_on_domain_bdr():
    init_triang = InitialTriangulation.unit_square()
    assert all(v.on_domain_boundary for v in init_triang.vertex_roots)
    init_triang.element_roots[0].refine()

    leaves = set(init_triang.element_roots)
    for _ in range(100):
        elem = leaves.pop()
        leaves.update(elem.refine())
    for vert in init_triang.vertex_meta_root.bfs():
        assert vert.on_domain_boundary == (vert.x == 0 or vert.x == 1
                                           or vert.y == 0 or vert.y == 1)


def test_vertex_subtree():
    import matplotlib.pyplot as plt
    T = InitialTriangulation.unit_square()
    elem_meta_root = T.elem_meta_root
    elem_meta_root.uniform_refine(6)

    vertex_subtree = MetaRootView.from_metaroot(T.vertex_meta_root)
    elem_subtree = elem_tree_from_vertex_tree(vertex_subtree)
    assert len(elem_subtree.bfs()) == 2

    # Create a subtree with only vertices lying below the diagonal.
    vertex_subtree.uniform_refine(1)
    vertex_subtree.local_refine(lambda vertex: vertex.x + vertex.y <= 1)
    assert len(vertex_subtree.bfs()) < len(T.vertex_meta_root.bfs())

    elem_subtree = elem_tree_from_vertex_tree(vertex_subtree)
    assert len(elem_subtree.bfs()) < len(T.elem_meta_root.bfs())


def test_vertex_tree():
    T = InitialTriangulation.unit_square()
    T.refine(T.elements[0])
    assert T.vertices[4].parents == [T.vertices[0], T.vertices[1]]
    assert T.vertices[0].children == [T.vertices[4]]
    assert T.vertices[1].children == [T.vertices[4]]

    T.refine(T.elements[2])
    assert T.vertices[5].on_domain_boundary
    assert T.vertices[5].parents == [T.vertices[4]]
    assert T.vertices[4].children == [T.vertices[5]]

    T.refine(T.elements[4])
    assert T.vertices[6].on_domain_boundary
    assert T.vertices[6].parents == [T.vertices[4]]
    assert T.vertices[4].children == [T.vertices[5], T.vertices[6]]

    T.refine(T.elements[6])
    assert not T.vertices[8].on_domain_boundary
    assert T.vertices[8].parents == [T.vertices[7], T.vertices[5]]


def test_vertex_patch():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(1)

    elements = T.elem_meta_root.bfs()
    vertices = T.vertex_meta_root.bfs()
    assert vertices[0].patch == [elements[0]]
    assert vertices[1].patch == [elements[1]]
    assert vertices[2].patch == elements[0:2]
    assert vertices[3].patch == elements[0:2]
    assert vertices[4].patch == elements[2:6]

    elements = set(elements)
    for _ in range(100):
        elem = elements.pop()
        if not elem.is_full():
            elements.update(elem.refine())
    vertices = T.vertex_meta_root.bfs()
    for v in vertices:
        if v.level == 0: continue
        if v.on_domain_boundary:
            assert len(v.patch) == 2
        else:
            assert len(v.patch) == 4


def test_unif_refinement():
    T = InitialTriangulation.unit_square()
    elem_meta_root = T.elem_meta_root
    assert elem_meta_root.is_full()
    for root in elem_meta_root.roots:
        assert root.level == 0

    elem_meta_root.uniform_refine(5)
    assert len(elem_meta_root.bfs()) == (2**6 - 1) * len(elem_meta_root.roots)
    counts = defaultdict(int)
    for elem in elem_meta_root.bfs():
        assert elem.level <= 5
        counts[elem.level] += 1

    for level in range(6):
        assert counts[level] == 2**level * len(elem_meta_root.roots)


def test_elem_tree():
    T = InitialTriangulation.unit_square()
    elem_meta_root = T.elem_meta_root
    assert elem_meta_root.is_full()
    for root in elem_meta_root.roots:
        assert root.level == 0

    elem_meta_root.uniform_refine(3)
    for elem in elem_meta_root.bfs():
        assert elem.level <= 3


def test_vertex_tree():
    T = InitialTriangulation.unit_square()
    vertex_meta_root = T.vertex_meta_root
    assert vertex_meta_root.is_full()
    for root in vertex_meta_root.roots:
        assert root.level == 0

    vertex_meta_root.uniform_refine(3)
    for vertex in vertex_meta_root.bfs():
        assert vertex.level <= 3
    assert len(
        T.elem_meta_root.bfs()) == (2**4 - 1) * len(T.elem_meta_root.roots)


def test_refined_tree():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
