import numpy as np

from .basis import HierarchicalBasisFunction
from .triangulation import InitialTriangulation


def test_hierarchical_basis():
    T = InitialTriangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    basis_meta_root.root.refine()
    assert basis_meta_root.root.is_full()
    for root in basis_meta_root.root.children[0]:
        assert root.is_full()
        root.refine()
    assert len(T.elem_meta_root.bfs()) == 2


def test_diamond_no_overrefine():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(2)
    # Find the first leaf and refine it, creating a diamond in the vertex tree.
    leaves = [elem for elem in T.elem_meta_root.bfs() if elem.is_leaf()]
    assert len(leaves) == 8
    leaves[0].refine()

    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    basis_meta_root.uniform_refine(0)

    # Refine and find the node view with the single vertex on level 1.
    level1_fn = basis_meta_root.root.children[0][0].refine()[0]
    assert level1_fn.level == 1

    # Refine *a single child* and find the node view on level 2.
    level2_fn = level1_fn.refine(children=[level1_fn.node.children[0]])[0]
    assert level2_fn.level == 2

    # Refine this new node view; the diamond should be refined but we should
    # *not* have the entire vertex tree.
    level2_fn.refine(make_conforming=True)
    assert len(basis_meta_root.bfs()) == 8
    assert len(T.vertex_meta_root.bfs()) == 10


def test_refine_hierarchical_basis():
    T = InitialTriangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    basis_meta_root.root.refine()
    for root in basis_meta_root.root.children[0]:
        assert root.is_full()
        root.node.refine()
        root.refine()
    assert len(T.elem_meta_root.bfs()) == 6
    assert len(T.vertex_meta_root.bfs()) == 5
    assert len(basis_meta_root.bfs()) == 5

    leaves = set([f for f in basis_meta_root.bfs() if f.is_leaf()])
    for i in range(800):
        f = leaves.pop()
        f.node.refine()
        leaves.update(f.refine(make_conforming=True))
        assert len(basis_meta_root.bfs()) == len(T.vertex_meta_root.bfs())


def test_eval_basis():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(6)
    basis = HierarchicalBasisFunction.from_triangulation(T)
    basis.deep_refine()
    for phi in basis.bfs():
        # phi should be zero outside the domain.
        assert np.allclose(phi.eval(np.array([[-1, 0], [1, -1], [2, 2]]).T), 0)
        assert np.isclose(phi.eval(np.array([3, 3])), 0)
        for elem in phi.support:
            # phi should be either 0 or 1 in the vertices of an element.
            verts = np.array([elem.vertices[i].as_array() for i in range(3)]).T
            v_eval = phi.eval(verts)
            assert np.allclose(v_eval.sum(), 1.0)
            assert np.allclose(v_eval, np.array(elem.vertices) == phi.node)

            # Take random combination of vertices.
            v_index = elem.vertices.index(phi.node)

            alpha = np.random.rand(3, 4)
            alpha /= np.sum(alpha, axis=0)
            assert np.allclose(phi.eval(verts @ alpha), alpha[v_index, :])
