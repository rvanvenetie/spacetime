from basis import HaarBasis, OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis
from applicator import Applicator
from index_set import IndexSet
from indexed_vector import IndexedVector

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pytest


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    for basis in [
            HaarBasis.uniform_basis(max_level=5),
            HaarBasis.origin_refined_basis(max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.singlescale_mass, Lambda)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda))
                vec = IndexedVector(Lambda, np_vec)
                res = applicator.apply(vec)
                assert np.allclose(
                    [(1.0 if index[0] == 0 else 2**(index[0] - 1)) * res[index]
                     for index in sorted(res.keys())], np_vec)


def test_orthonormal_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    for basis in [
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=5),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.singlescale_mass, Lambda)
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


def test_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    ml = 4
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            HaarBasis.origin_refined_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml),
            ThreePointBasis.origin_refined_basis(max_level=ml)
    ]:
        for l in range(1, ml + 1):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.singlescale_mass, Lambda)
            eye = np.eye(len(Lambda))
            for i, labda in enumerate(sorted(Lambda)):
                supp_labda = basis.wavelet_support(labda)
                vec = IndexedVector(Lambda, eye[i, :])
                res = applicator.apply(vec)
                for j, mu in enumerate(sorted(Lambda)):
                    supp_mu = basis.wavelet_support(mu)
                    supp_total = supp_labda.intersection(supp_mu)
                    true_val = 0.0
                    if supp_total:
                        true_val = quad(lambda x: basis.eval_wavelet(labda, x)
                                        * basis.eval_wavelet(mu, x),
                                        supp_total.a,
                                        supp_total.b,
                                        points=[
                                            supp_labda.a, supp_labda.mid,
                                            supp_labda.b, supp_mu.a,
                                            supp_mu.mid, supp_mu.b,
                                            supp_total.a, supp_total.mid,
                                            supp_total.b
                                        ])[0]
                    try:
                        assert np.isclose(res[mu], true_val)
                    except AssertionError:
                        #xx = np.linspace(0, 1, 1025)
                        #plt.plot(xx, basis.eval_wavelet(labda, xx))
                        #plt.plot(xx, basis.eval_wavelet(mu, xx))
                        #plt.show()
                        print('basis=', basis, 'level=', l, 'Lambda=', Lambda,
                              'labda=', labda, 'mu=', mu, 'found val=',
                              res[mu], 'true val=', true_val, 'supp=',
                              supp_total)
                        #raise
    assert False


def test_orthonormal_multiscale_damping_correct():
    """ Test that the multiscale damping matrix is correct. """
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

    basis = OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=5)
    for l in range(1, 3):
        Lambda = basis.indices.until_level(l)
        applicator = Applicator(basis, basis.singlescale_damping, Lambda)
        eye = np.eye(len(Lambda))
        for i in range(len(Lambda)):
            vec = IndexedVector(Lambda, eye[i, :])
            assert np.allclose(
                applicator.apply(vec).asarray(),
                reference_damping_matrix[:len(Lambda), i])


def test_apply_upp_low_vs_full():
    """ Test that apply_upp() + apply_low() == apply(). """
    for basis in [
            HaarBasis.uniform_basis(max_level=5),
            HaarBasis.origin_refined_basis(max_level=5),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=5),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=5),
            ThreePointBasis.uniform_basis(max_level=5),
            ThreePointBasis.origin_refined_basis(max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.singlescale_mass, Lambda)
            for _ in range(10):
                c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
                res_full_op = applicator.apply(c)
                res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
                assert np.allclose(res_full_op.asarray(),
                                   res_upp_low.asarray())
