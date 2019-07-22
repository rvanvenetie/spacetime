from basis import HaarBasis, OrthonormalDiscontinuousLinearBasis
from index import IndexSet, IndexedVector
import numpy as np
import pytest


def test_haar_singlescale_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    b = HaarBasis()
    for l in range(1, 5):
        indices = b.scaling_indices_on_level(l)
        assert len(indices.indices) == 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**l))
            res = b.singlescale_mass(l=l, Pi=indices, Pi_A=indices, d=d)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_orthonormal_hierarchy():
    b = OrthonormalDiscontinuousLinearBasis()
    assert b.wavelet_indices_on_level(1).indices == {(1, 0), (1, 1)}

    assert b.scaling_parents((1, 0)) == [(0, 0), (0, 1)]
    assert b.scaling_parents((1, 1)) == [(0, 0), (0, 1)]
    assert b.scaling_parents((1, 2)) == [(0, 0), (0, 1)]
    assert b.scaling_parents((1, 3)) == [(0, 0), (0, 1)]

    assert b.scaling_parents((2, 0)) == [(1, 0), (1, 1)]
    assert b.scaling_parents((2, 1)) == [(1, 0), (1, 1)]
    assert b.scaling_parents((2, 2)) == [(1, 0), (1, 1)]
    assert b.scaling_parents((2, 3)) == [(1, 0), (1, 1)]
    assert b.scaling_parents((2, 4)) == [(1, 2), (1, 3)]
    assert b.scaling_parents((2, 5)) == [(1, 2), (1, 3)]
    assert b.scaling_parents((2, 6)) == [(1, 2), (1, 3)]
    assert b.scaling_parents((2, 7)) == [(1, 2), (1, 3)]

    assert b.scaling_children((0, 0)) == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert b.scaling_children((0, 1)) == [(1, 0), (1, 1), (1, 2), (1, 3)]

    assert b.scaling_children((1, 0)) == [(2, 0), (2, 1), (2, 2), (2, 3)]
    assert b.scaling_children((1, 1)) == [(2, 0), (2, 1), (2, 2), (2, 3)]
    assert b.scaling_children((1, 2)) == [(2, 4), (2, 5), (2, 6), (2, 7)]
    assert b.scaling_children((1, 3)) == [(2, 4), (2, 5), (2, 6), (2, 7)]

    assert b.scaling_siblings((2, 0)) == [(2, 0), (2, 1)]
    assert b.scaling_siblings((2, 1)) == [(2, 0), (2, 1)]
    assert b.scaling_siblings((2, 2)) == [(2, 0), (2, 1)]
    assert b.scaling_siblings((2, 3)) == [(2, 0), (2, 1)]

    assert b.wavelet_siblings((2, 0)) == [(2, i) for i in range(4)]
    assert b.wavelet_siblings((2, 1)) == [(2, i) for i in range(4)]
    assert b.wavelet_siblings((2, 2)) == [(2, i) for i in range(4, 8)]
    assert b.wavelet_siblings((2, 3)) == [(2, i) for i in range(4, 8)]


def test_orthonormal_singlescale_mass():
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 5):
        indices = b.scaling_indices_on_level(l)
        assert len(indices.indices) == 2 * 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2 * 2**l))
            res = b.singlescale_mass(l=l, Pi=indices, Pi_A=indices, d=d)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_orthonormal_singlescale_damping():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    b = OrthonormalDiscontinuousLinearBasis()
    indices = b.scaling_indices_on_level(0)

    d = IndexedVector(indices, [0.46, 0])
    res = b.singlescale_damping(l=0, Pi=indices, Pi_A=indices, d=d)
    assert np.allclose(res.asarray(), [0, 0])

    d = IndexedVector(indices, [0, 0.12])
    res = b.singlescale_damping(l=0, Pi=indices, Pi_A=indices, d=d)
    assert np.allclose(res.asarray(), [0.12 * 2 * np.sqrt(3), 0])
