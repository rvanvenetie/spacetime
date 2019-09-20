from .basis import HierarchicalBasisFunction
from .triangulation import Triangulation


def test_hierarchical_basis():
    T = Triangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    assert basis_meta_root.is_full()
    for root in basis_meta_root.roots:
        assert root.is_full()
        root.refine()
    assert len(T.elements) == 2


def test_refine_hierarchical_basis():
    T = Triangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    for root in basis_meta_root.roots:
        assert root.is_full()
        root.node.refine()
        root.refine()
    assert len(T.elements) == 6
    assert len(T.vertices) == 5
    assert len(basis_meta_root.bfs()) == 5

    leaves = set([f for f in basis_meta_root.bfs() if f.is_leaf()])
    for i in range(400):
        f = leaves.pop()
        f.node.refine()
        leaves.update(f.refine())
        assert len(basis_meta_root.bfs()) == len(T.vertices)
