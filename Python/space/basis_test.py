from .basis import HierarchicalBasisFunction
from .triangulation import Triangulation


def test_hierarhical_basis():
    T = Triangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    assert basis_meta_root.is_full()
    for root in basis_meta_root.roots:
        assert root.is_full()
        root.refine()
    assert len(T.elements) == 2


def test_refine_hierarhical_basis():
    T = Triangulation.unit_square()
    basis_meta_root = HierarchicalBasisFunction.from_triangulation(T)
    for root in basis_meta_root.roots:
        assert root.is_full()
        root.refine(refine_underlying_tree=True)
    assert len(T.elements) == 6
    assert len(T.vertices) == 5
    assert len(basis_meta_root.bfs()) == 5

    leaves = set([f for f in basis_meta_root.bfs() if f.is_leaf()])
    for i in range(50):
        f = leaves.pop()
        leaves.update(
            f.refine(make_conforming=True, refine_underlying_tree=True))
        assert len(basis_meta_root.bfs()) == len(T.vertices)
