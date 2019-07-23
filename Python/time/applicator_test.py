from basis import HaarBasis, OrthonormalDiscontinuousLinearBasis
from applicator import Applicator
from index_set import IndexSet
from indexed_vector import IndexedVector
import numpy as np
import pytest


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    b = HaarBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_mass, Lambda)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda))
                vec = IndexedVector(Lambda, np_vec)
                res = applicator.apply(vec)
                assert np.allclose(
                    [(1.0 if index[0] == 0 else 2**(index[0] - 1)) * res[index]
                     for index in sorted(res.keys())], np_vec)


def test_haar_apply_upp_low_vs_full():
    b = HaarBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_mass, Lambda)
            for _ in range(10):
                c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
                res_full_op = applicator.apply(c)
                res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
                assert np.allclose(res_full_op.asarray(),
                                   res_upp_low.asarray())


def test_orthonormal_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_mass, Lambda)
            eye = np.eye(len(Lambda))
            res_matrix = np.zeros([len(Lambda), len(Lambda)])
            for i in range(len(Lambda)):
                vec = IndexedVector(Lambda, eye[i, :])
                res = applicator.apply(vec)
                res_matrix[:, i] = res.asarray()
            assert np.allclose(eye, res_matrix)

            for _ in range(10):
                vec = np.random.rand(len(Lambda))
                res = applicator.apply(IndexedVector(Lambda, vec))
                assert np.allclose(vec, res.asarray())


def test_orthonormal_multiscale_damping_linear():
    """ Test that applying the multiscale damping operator is linear. """
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_damping, Lambda)
            for _ in range(10):
                v1 = np.random.rand(len(Lambda))
                v2 = np.random.rand(len(Lambda))
                v3 = v1 + v2
                assert np.allclose(
                    applicator.apply(IndexedVector(Lambda, v1)).asarray() +
                    applicator.apply(IndexedVector(Lambda, v2)).asarray(),
                    applicator.apply(IndexedVector(Lambda, v3)).asarray())
                assert np.allclose(
                    applicator.apply_low(IndexedVector(Lambda, v1)).asarray() +
                    applicator.apply_low(IndexedVector(Lambda, v2)).asarray(),
                    applicator.apply_low(IndexedVector(Lambda, v3)).asarray())
                assert np.allclose(
                    applicator.apply_upp(IndexedVector(Lambda, v1)).asarray() +
                    applicator.apply_upp(IndexedVector(Lambda, v2)).asarray(),
                    applicator.apply_upp(IndexedVector(Lambda, v3)).asarray())


def test_orthonormal_multiscale_damping_equivalent():
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 3):
        Lambda = b.uniform_wavelet_indices(max_level=l)
        applicator = Applicator(b, b.singlescale_damping, Lambda)
        eye = np.eye(len(Lambda))
        res_matrix = np.zeros([len(Lambda), len(Lambda)])
        res_matrix_ul = np.zeros([len(Lambda), len(Lambda)])
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            res = applicator.apply(vec)
            res_ul = applicator.apply_upp(vec) + applicator.apply_low(vec)
            res_matrix[:, i] = res.asarray()
            res_matrix_ul[:, i] = res_ul.asarray()
            assert np.allclose(res.asarray(), res_ul.asarray())

        vec = IndexedVector(Lambda, np.ones(len(Lambda)))
        res = applicator.apply(vec)
        res_ul = applicator.apply_upp(vec) + applicator.apply_low(vec)
        assert np.allclose(np.sum(res_matrix, axis=1), res.asarray())
        assert np.allclose(res.asarray(), res_ul.asarray())


def test_orthonormal_multiscale_damping_correct():
    # Computed with Mathematica.
    sq3 = np.sqrt(3)
    sq2 = np.sqrt(2)
    sq6 = np.sqrt(6)
    reference_damping_matrix = np.array(
        [[0, 2 * sq3, -6, 0, -6 * sq2, 0, -6 * sq2, 0],
         [0, 0, 0, 6, 3 * sq6, 3 * sq2, -3 * sq6, 3 * sq2],
         [0, 0, 0, 2 * sq3, 3 * sq2, -3 * sq6, -3 * sq2, -3 * sq6],
         [0, 0, 0, 0, 0, -6 * sq2, 0, 6 * sq2], [0, 0, 0, 0, 0, 4 * sq3, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4 * sq3],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 3):
        Lambda = b.uniform_wavelet_indices(max_level=l)
        applicator = Applicator(b, b.singlescale_damping, Lambda)
        eye = np.eye(len(Lambda))
        res_matrix = np.zeros([len(Lambda), len(Lambda)])
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            res = applicator.apply_low(vec) + applicator.apply_upp(vec)
            res_matrix[:, i] = res.asarray()
        assert np.allclose(
            res_matrix, reference_damping_matrix[:len(Lambda), :len(Lambda)])
        vec = IndexedVector(Lambda, np.ones(len(Lambda)))
        res = applicator.apply(vec)
        assert np.allclose(np.sum(res_matrix, axis=1), res.asarray())
