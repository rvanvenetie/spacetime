from random import random

from ..space.triangulation import InitialTriangulation
from ..time.haar_basis import HaarBasis
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .multi_tree_view import MultiNodeView, MultiTree
from .tree_view import TreeView


class TripleNodeView(MultiNodeView):
    dim = 3


class TripleTreeView(MultiTree):
    mlt_node_cls = TripleNodeView


def test_multi_tree_view():
    basis_haar = HaarBasis()
    basis_haar.metaroot_wavelet.uniform_refine(5)
    metaroot_haar = basis_haar.metaroot_wavelet
    basis_nodes = metaroot_haar.bfs()

    multi_tree = MultiTree.from_metaroots(
        (metaroot_haar, metaroot_haar, metaroot_haar),
        mlt_node_cls=TripleNodeView)
    multi_tree.deep_refine()

    assert len(multi_tree.bfs()) == len(basis_nodes)**3

    multi_tree = MultiTree.from_metaroots(
        (metaroot_haar, metaroot_haar, metaroot_haar),
        mlt_node_cls=TripleNodeView)
    multi_tree.sparse_refine(3)

    assert len(multi_tree.bfs()) < len(basis_nodes)**2

    multi_tree_copy = MultiTree(
        TripleNodeView((metaroot_haar, metaroot_haar, metaroot_haar)))
    multi_tree_copy.union(multi_tree)

    assert len(multi_tree.bfs()) == len(multi_tree_copy.bfs())


def test_multi_tree_bfs_kron():
    metaroot_haar = HaarBasis.metaroot_wavelet
    metaroot_ortho = OrthonormalBasis.metaroot_wavelet
    metaroot_three = ThreePointBasis.metaroot_wavelet
    for metaroot in [metaroot_haar, metaroot_ortho, metaroot_three]:
        metaroot.uniform_refine(5)

    # Create sparse multi tree.
    vec_sparse = MultiTree.from_metaroots(
        (metaroot_haar, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeView)
    vec_sparse.sparse_refine(3)

    # Create uniform multi tree.
    vec_unif = MultiTree.from_metaroots(
        (metaroot_haar, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeView)
    vec_unif.uniform_refine([1, 5, 2])

    assert len(vec_sparse.bfs()) == len(vec_sparse.bfs_kron())
    assert len(vec_unif.bfs()) == len(vec_unif.bfs_kron())

    # Verify that ordering of the dofs is indeed according to `kron`.
    for bfs_kron in [vec_sparse.bfs_kron(), vec_unif.bfs_kron()]:
        for i in range(1, len(bfs_kron)):
            labdas_prev = tuple(bfs_kron[i - 1].nodes[j].labda
                                for j in range(3))
            labdas_now = tuple(bfs_kron[i].nodes[j].labda for j in range(3))

            assert labdas_now[0] >= labdas_prev[0]
            for j in range(3):
                if labdas_now[j] == labdas_prev[j]:
                    continue
                assert labdas_now[j] > labdas_prev[j]
                break

    # Verify that it also coincides with the normal bfs order for trees.
    tree = TreeView.from_metaroot(metaroot_three)
    tree.deep_refine()
    assert tree.bfs() == tree.bfs_kron()


def test_multi_tree_make_conforming():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
    metaroot_ortho = OrthonormalBasis.metaroot_wavelet
    metaroot_three = ThreePointBasis.metaroot_wavelet
    for metaroot in [metaroot_ortho, metaroot_three]:
        metaroot.uniform_refine(5)

    # Create uniform multi tree.
    tree_unif = TripleTreeView.from_metaroots(
        (T.vertex_meta_root, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeView)
    tree_unif.uniform_refine(3)

    # Mark the multinodes all having level 2
    nodes = tree_unif.bfs()
    marked_nodes = list(
        filter(lambda mlt_node: all(n.level == 2 for n in mlt_node.nodes),
               nodes))

    # Create new tree with these marked nodes.
    tree_lvl_2 = TripleTreeView.make_conforming(marked_nodes)

    # Check that it only contains nodes having max level <= 2.
    for mlt_node in tree_lvl_2.bfs():
        assert max(n.level for n in mlt_node.nodes) <= 2

    # Check that the new tree contains all marked nodes.
    nodes_lvl_2 = set(tuple(n.nodes) for n in tree_lvl_2.bfs())
    for n in marked_nodes:
        assert tuple(n.nodes) in nodes_lvl_2

    # Again, but now mark some random nodes.
    nodes = tree_unif.bfs()
    random_nodes = [node for node in nodes if random() < 0.2]

    # Create new tree with these marked nodes.
    tree_random = TripleTreeView.make_conforming(random_nodes)

    # Check that the new tree contains all marked nodes.
    nodes_random = set(tuple(n.nodes) for n in tree_random.bfs())
    for n in random_nodes:
        assert tuple(n.nodes) in nodes_random
