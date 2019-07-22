from basis import HaarBasis, OrthonormalDiscontinuousLinearBasis
from index import IndexSet, IndexedVector
import numpy as np


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


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    b = HaarBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            eye = np.eye(len(Lambda))
            res_matrix = eye
            for i in range(len(Lambda)):
                vec = IndexedVector(Lambda, eye[:, i])
                res = b.apply_operator(b.singlescale_mass, Lambda, vec)
                assert sum([
                    (1.0 if index[0] == 0 else 2**(index[0] - 1)) * res[index]
                    for index in res.vector
                ]) == 1.0
                res_matrix[:, i] = res.asarray()


def test_haar_apply_P_Q():
    """ Test that the application of the P and Q matrices are as expected. """
    b = HaarBasis()
    for l in range(1, 5):
        Delta_l = b.scaling_indices_on_level(l)
        Delta_lm1 = b.scaling_indices_on_level(l - 1)
        Lambda_l = b.wavelet_indices_on_level(l)
        for _ in range(100):
            vec = IndexedVector(Delta_lm1, np.random.rand(2**(l - 1)))
            res = b.apply_P(l, Delta_lm1, Delta_l, vec)
            assert np.allclose(res.asarray(), np.kron(vec.asarray(), [1, 1]))

            vec = IndexedVector(Delta_l, np.random.rand(2**l))
            res = b.apply_PT(l, Delta_l, Delta_lm1, vec)
            assert np.allclose(res.asarray(),
                               vec.asarray()[::2] + vec.asarray()[1::2])

            vec = IndexedVector(Lambda_l, np.random.rand(2**(l - 1)))
            res = b.apply_Q(l, Lambda_l, Delta_l, vec)
            assert np.allclose(res.asarray(), np.kron(vec.asarray(), [1, -1]))

            vec = IndexedVector(Delta_l, np.random.rand(2**l))
            res = b.apply_QT(l, Delta_l, Lambda_l, vec)
            assert np.allclose(res.asarray(),
                               vec.asarray()[::2] - vec.asarray()[1::2])


def test_haar_apply_upp_low_vs_full():
    b = HaarBasis()
    for l in range(1, 6):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
            operator = b.singlescale_mass
            res_full_op = b.apply_operator(operator, Lambda, c)
            res_upp_low = IndexedVector.sum(
                b.apply_operator_low(operator, Lambda, c),
                b.apply_operator_upp(operator, Lambda, c))
            assert np.allclose(res_full_op.asarray(), res_upp_low.asarray())


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


def test_orthonormal_multiscale_mass():
    b = OrthonormalDiscontinuousLinearBasis()
    for l in range(1, 7):
        for Lambda in [
                b.uniform_wavelet_indices(max_level=l),
                b.origin_refined_wavelet_indices(max_level=l)
        ]:
            eye = np.eye(len(Lambda))
            res_matrix = eye
            for i in range(len(Lambda)):
                vec = IndexedVector(Lambda, eye[i, :])
                res = b.apply_operator(b.singlescale_mass, Lambda, vec)
                res_matrix[:, i] = res.asarray()
            assert np.allclose(eye, res_matrix)


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


def test_orthonormal_multiscale_damping():
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
        eye = np.eye(len(Lambda))
        res_matrix = eye
        res_matrix_ul = eye
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            res = b.apply_operator(b.singlescale_damping, Lambda, vec)
            res_ul = IndexedVector.sum(
                b.apply_operator_low(b.singlescale_damping, Lambda, vec),
                b.apply_operator_upp(b.singlescale_damping, Lambda, vec))
            #res_matrix_ul[:, i] = res_ul.asarray()
            res_matrix[:, i] = res_ul.asarray()
            #assert np.allclose(res.asarray(), res_ul.asarray())
        np.set_printoptions(linewidth=10000)
        print(res_matrix /
              reference_damping_matrix[:len(Lambda), :len(Lambda)])
    assert np.allclose(res_matrix,
                       reference_damping_matrix[:len(Lambda), :len(Lambda)])
