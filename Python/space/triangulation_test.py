import numpy as np

from .triangulation import Triangulation


def test_on_domain_bdr():
    triangulation = Triangulation.unit_square()
    assert all([v.on_domain_boundary for v in triangulation.vertices])
    triangulation.refine(triangulation.elements[0])
    for _ in range(100):
        triangulation.refine(triangulation.elements[np.random.randint(
            len(triangulation.vertices))])
    for vert in triangulation.vertices:
        assert vert.on_domain_boundary == (vert.x == 0 or vert.x == 1
                                           or vert.y == 0 or vert.y == 1)


def test_vertex_tree():
    T = Triangulation.unit_square()
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
    T = Triangulation.unit_square()
    T.refine(T.elements[0])
    assert T.vertices[0].patch == [T.elements[0]]
    assert T.vertices[1].patch == [T.elements[1]]
    assert T.vertices[2].patch == T.elements[0:2]
    assert T.vertices[3].patch == T.elements[0:2]
    assert T.vertices[4].patch == T.elements[2:6]
    for _ in range(100):
        T.refine(T.elements[np.random.randint(len(T.vertices))])
    for v in T.vertices:
        if v.level == 0: continue
        if v.on_domain_boundary:
            assert len(v.patch) == 2
        else:
            assert len(v.patch) == 4


def test_elem_tree():
    T = Triangulation.unit_square()
    elem_meta_root = T.elem_meta_root
    assert elem_meta_root.is_full()
    for root in elem_meta_root.roots:
        assert root.level == 0

    elem_meta_root.uniform_refine(3)
    assert set(elem_meta_root.bfs()) == set(T.elements)

    for elem in T.elements:
        assert elem.level <= 3


def test_vertex_tree():
    T = Triangulation.unit_square()
    vertex_meta_root = T.vertex_meta_root
    assert vertex_meta_root.is_full()
    for root in vertex_meta_root.roots:
        assert root.level == 0

    vertex_meta_root.uniform_refine(3)
    assert set(vertex_meta_root.bfs()) == set(T.vertices)
    for vertex in T.vertices:
        assert vertex.level <= 3
    assert len(T.elements) == (2**4 - 1) * len(T.elem_meta_root.roots)


def test_refined_tree():
    T = Triangulation.unit_square()
    for _ in range(5):
        T.refine_uniform()
