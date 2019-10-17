import random
from collections import defaultdict

from ..time.haar_basis import HaarBasis
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .multi_tree_vector import MultiNodeVector, MultiTreeVector


class TripleNodeVector(MultiNodeVector):
    dim = 3


def test_multi_tree_view():
    metaroot_haar = HaarBasis.metaroot_wavelet
    metaroot_ortho = OrthonormalBasis.metaroot_wavelet
    metaroot_three = ThreePointBasis.metaroot_wavelet
    for metaroot in [metaroot_haar, metaroot_ortho, metaroot_three]:
        metaroot.uniform_refine(5)

    # Fill sparse vector with random junk.
    vec_sparse = MultiTreeVector.from_metaroots(
        (metaroot_haar, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeVector)
    vec_sparse.sparse_refine(
        3, call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    # Fill another uniform vector with random junk.
    vec_unif = MultiTreeVector.from_metaroots(
        (metaroot_haar, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeVector)
    vec_unif.uniform_refine(
        [1, 5, 2],
        call_postprocess=lambda nv: setattr(nv, 'value', random.random()))

    # Create empty vector.
    vec_result = MultiTreeVector.from_metaroots(
        (metaroot_haar, metaroot_ortho, metaroot_three),
        mlt_node_cls=TripleNodeVector)

    # vec_result = vec_sparse * 3.14 + vec_unif * 1337
    vec_result.axpy(vec_sparse, 3.14)
    vec_result.axpy(vec_unif, 1337)

    # Verify that this is correct using a dictionary.
    vec_dict = defaultdict(float)
    for nv in vec_sparse.bfs():
        vec_dict[tuple(nv.nodes)] = 3.14 * nv.value
    for nv in vec_unif.bfs():
        vec_dict[tuple(nv.nodes)] += 1337 * nv.value

    for nv in vec_result.bfs():
        assert nv.value == vec_dict[tuple(nv.nodes)]
