from ..datastructures.tree_view import MetaRootView, NodeView
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView


def test_vertex_subtree():
    T = InitialTriangulation.unit_square()
    elem_meta_root = T.elem_meta_root
    elem_meta_root.uniform_refine(6)

    vertex_subtree = MetaRootView.from_metaroot(T.vertex_meta_root)
    T_view = TriangulationView(vertex_subtree)
    assert len(T_view.elements) == 2

    # Create a subtree with only vertices lying below the diagonal.
    vertex_subtree = MetaRootView.from_metaroot_deep(
        T.vertex_meta_root,
        call_filter=lambda vertex: vertex.x + vertex.y <= 1)
    assert len(vertex_subtree.bfs()) < len(T.vertex_meta_root.bfs())
    T_view = TriangulationView(vertex_subtree)
    assert len(T_view.elements) < len(T.elem_meta_root.bfs())

    # Check that the history object contains exactly non-root vertices
    assert len(T_view.history) == len(vertex_subtree.bfs()) - len(
        vertex_subtree.roots)

    # Check that we do not have duplicates
    vertices_subtree = set(v.node for v in vertex_subtree.bfs())
    assert len(vertex_subtree.bfs()) == len(vertices_subtree)

    # Check all nodes necessary for the elem subtree are inside the vertices_subtree
    for elem in T_view.elements:
        for vtx in elem.node.vertices:
            assert vtx in vertices_subtree

    # And the other way around.
    elements_subtree = set(e.node for e in T_view.elements)
    assert len(T_view.elements) == len(elements_subtree)
    for vertex in vertex_subtree.bfs():
        for elem in vertex.node.patch:
            assert elem in elements_subtree
