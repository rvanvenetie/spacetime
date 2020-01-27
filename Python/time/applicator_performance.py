import numpy as np

from ..datastructures.tree_view import TreeView
from . import operators
from .applicator import Applicator
from .orthonormal_basis import OrthonormalBasis
from .sparse_vector import SparseVector

seed = 0


def bsd_rnd():
    global seed
    seed = (1103515245 * seed + 12345) & 0x7fffffff
    return seed


def test_python(level, bilform_iters, inner_iters):
    basis, _ = OrthonormalBasis.uniform_basis(max_level=level)

    for _ in range(bilform_iters):
        vec = TreeView.from_metaroot(OrthonormalBasis.metaroot_wavelet)
        vec.deep_refine(
            call_filter=lambda fn: fn.level <= 0 or (bsd_rnd() % 3) != 0)
        Lambda = [node.node for node in vec.bfs()]
        applicator = Applicator(operators.mass(basis), basis)
        for _ in range(inner_iters):
            np_vec = np.ones(len(Lambda))
            vec = SparseVector(Lambda, np_vec)
            applicator.apply(vec)


if __name__ == "__main__":
    test_python(level=10, bilform_iters=10, inner_iters=150)
