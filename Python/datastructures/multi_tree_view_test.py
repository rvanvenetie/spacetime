from ..time.haar_basis import HaarBasis
from .multi_tree_view import MultiNodeView, MultiTree


class TripleNodeView(MultiNodeView):
    dim = 3


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
    multi_tree.root._sparse_refine(3)

    assert len(multi_tree.bfs()) < len(basis_nodes)**2

    multi_tree_copy = MultiTree(
        TripleNodeView((metaroot_haar, metaroot_haar, metaroot_haar)))
    multi_tree_copy.union(multi_tree)

    assert len(multi_tree.bfs()) == len(multi_tree_copy.bfs())
