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
            eye = np.eye(len(Lambda))
            res_matrix = eye
            for i in range(len(Lambda)):
                vec = IndexedVector(Lambda, eye[:, i])
                res = applicator.apply(vec)
                assert sum([
                    (1.0 if index[0] == 0 else 2**(index[0] - 1)) * res[index]
                    for index in res.vector
                ]) == 1.0
                res_matrix[:, i] = res.asarray()


def test_haar_apply_upp_low_vs_full():
    b = HaarBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_mass, Lambda)
            c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
            res_full_op = applicator.apply(c)
            res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
            assert np.allclose(res_full_op.asarray(), res_upp_low.asarray())


def test_orthonormal_multiscale_mass():
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 7):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            applicator = Applicator(b, b.singlescale_mass, Lambda)
            eye = np.eye(len(Lambda))
            res_matrix = eye
            for i in range(len(Lambda)):
                vec = IndexedVector(Lambda, eye[i, :])
                res = applicator.apply(vec)
                res_matrix[:, i] = res.asarray()
            assert np.allclose(eye, res_matrix)


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
        res_matrix = eye
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            res = applicator.apply_low(vec) + applicator.apply_upp(vec)
            res_matrix[:, i] = res.asarray()
    assert np.allclose(res_matrix,
                       reference_damping_matrix[:len(Lambda), :len(Lambda)])


@pytest.mark.skip(reason="Magically does not work")
def test_orthonormal_multiscale_damping_equivalent():
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 3):
        Lambda = b.uniform_wavelet_indices(max_level=l)
        applicator = Applicator(b, b.singlescale_damping, Lambda)
        eye = np.eye(len(Lambda))
        res_matrix = eye
        res_matrix_ul = eye
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            res = applicator.apply(vec)
            res_ul = applicator.apply_upp(vec) + applicator.apply_low(vec)
            res_matrix[:, i] = res.asarray()
            res_matrix_ul[:, i] = res_ul.asarray()
            #assert np.allclose(res.asarray(), res_ul.asarray())
        np.set_printoptions(linewidth=10000)
        print(res_matrix /
              reference_damping_matrix[:len(Lambda), :len(Lambda)])
    assert False
