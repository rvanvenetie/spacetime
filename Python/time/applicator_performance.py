import numpy as np

from . import operators
from ..datastructures.tree_vector import TreeVector
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
        vec_in = TreeVector.from_metaroot(OrthonormalBasis.metaroot_wavelet)
        vec_out = TreeVector.from_metaroot(OrthonormalBasis.metaroot_wavelet)
        vec_in.deep_refine(
            call_filter=lambda fn: fn.level <= 0 or (bsd_rnd() % 3) != 0)
        vec_out.deep_refine(
            call_filter=lambda fn: fn.level <= 0 or (bsd_rnd() % 3) != 0)
        applicator = Applicator(operators.mass(basis), basis)
        for _ in range(inner_iters):
            for nv in vec_in.bfs():
                nv.value = bsd_rnd()
            applicator.apply_low(vec_in, vec_out)
            vec_out.reset()
            applicator.apply_upp(vec_in, vec_out)
            vec_out.reset()


if __name__ == "__main__":
    test_python(level=10, bilform_iters=10, inner_iters=150)
