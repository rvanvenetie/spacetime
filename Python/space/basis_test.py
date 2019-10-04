import numpy as np

from .basis import HierarchicalBasisFunction
from .triangulation import InitialTriangulation


def test_hierarchical_basis():
    T = InitialTriangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    assert basis_meta_root.is_full()
    for root in basis_meta_root.roots:
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

    # Refine and find the node view with the single vertex on level 1.
    level1_fn = basis_meta_root.roots[0].refine()[0]
    assert level1_fn.level == 1

    # Refine *a single child* and find the node view on level 2.
    level2_fn = level1_fn.refine(children=[level1_fn.node.children[0]])[0]
    assert level2_fn.level == 2

    # Refine this new node view; the diamond should be refined but we should
    # *not* have the entire vertex tree.
    level2_fn.refine()
    assert len(basis_meta_root.bfs()) == 8
    assert len(T.vertex_meta_root.bfs()) == 10


def test_refine_hierarchical_basis():
    T = InitialTriangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    for root in basis_meta_root.roots:
        assert root.is_full()
        root.node.refine()
        root.refine()
    assert len(T.elem_meta_root.bfs()) == 6
    assert len(T.vertex_meta_root.bfs()) == 5
    assert len(basis_meta_root.bfs()) == 5

    leaves = set([f for f in basis_meta_root.bfs() if f.is_leaf()])
    for i in range(400):
        f = leaves.pop()
        f.node.refine()
        leaves.update(f.refine())
        assert len(basis_meta_root.bfs()) == len(T.vertex_meta_root.bfs())


def test_eval_basis():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(6)
    basis = HierarchicalBasisFunction.from_triangulation(T)
    basis.deep_refine()
    for phi in basis.bfs():
        assert phi.eval([-1, -1]) == 0
        assert phi.eval([-1, 0]) == 0
        for elem in phi.support:
            verts = [elem.vertices[i].as_array() for i in range(3)]
            v_eval = [phi.eval(verts[i]) for i in range(3)]
            assert np.allclose(sum(v_eval), 1.0)
            for i in range(3):
                if elem.vertices[i] == phi.node:
                    assert v_eval[i] == 1.0
                else:
                    assert v_eval[i] == 0.0
            # Take random combiniation of vertices
            alpha = np.random.rand(3)
            alpha /= sum(alpha)
            p = alpha[0] * verts[0] + alpha[1] * verts[1] + alpha[2] * verts[2]
            phi_eval = phi.eval(p)
            v_index = elem.vertices.index(phi.node)
            assert np.allclose(phi_eval, alpha[v_index])
