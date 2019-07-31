from applicator import Applicator

from haar_basis import HaarBasis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis, ms2ss, ss2ms, position_ms
from indexed_vector import IndexedVector

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pytest

np.random.seed(0)
np.set_printoptions(linewidth=10000, precision=3)


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    for basis in [
            HaarBasis.uniform_basis(max_level=5),
            HaarBasis.origin_refined_basis(max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.singlescale_mass(), Lambda)
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
            applicator = Applicator(basis, basis.singlescale_mass(), Lambda)
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


def test_multiscale_operator_quadrature():
    """ Test that the multiscale matrix equals that found with quadrature. """
    ml = 4
    hbu = HaarBasis.uniform_basis(max_level=ml)
    hbo = HaarBasis.origin_refined_basis(max_level=ml)
    oru = OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml)
    oro = OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
        max_level=ml)
    tpu = ThreePointBasis.uniform_basis(max_level=ml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=ml)
    for basis_in, basis_out, operator, deriv in [
        (hbu, hbu, hbu.singlescale_mass(hbu), (False, False)),
        (hbo, hbo, hbo.singlescale_mass(hbo), (False, False)),
        (hbu, hbo, hbu.singlescale_mass(hbo), (False, False)),
        (oru, oru, oru.singlescale_mass(oru), (False, False)),
        (oro, oro, oro.singlescale_mass(oro), (False, False)),
        (oru, oro, oru.singlescale_mass(oro), (False, False)),
        (oru, oru, oru.singlescale_damping(oru), (True, False)),
        (oro, oro, oro.singlescale_damping(oro), (True, False)),
        (oru, oro, oru.singlescale_damping(oro), (True, False)),
        (tpu, tpu, tpu.singlescale_mass(tpu), (False, False)),
        (tpo, tpo, tpo.singlescale_mass(tpo), (False, False)),
            # 3-point mass currently only works for the same in- and out basis.
            #(tpu, tpo, tpu.singlescale_mass(tpo), (False, False)),
        (tpu, tpu, tpu.singlescale_damping(tpu), (True, False)),
        (tpo, tpo, tpo.singlescale_damping(tpo), (True, False)),
            #(tpu, tpo, tpu.singlescale_damping(tpo), (True, False)),
    ]:
        for l in range(1, ml + 1):
            Lambda_in = basis_in.indices.until_level(l)
            Lambda_out = basis_out.indices.until_level(l)
            applicator = Applicator(basis_in, operator, Lambda_in, basis_out,
                                    Lambda_out)
            eye = np.eye(len(Lambda_in))
            resmat = np.zeros([len(Lambda_in), len(Lambda_out)])
            truemat = np.zeros([len(Lambda_in), len(Lambda_out)])
            for i, labda in enumerate(sorted(Lambda_in)):
                supp_labda = basis_in.wavelet_support(labda)
                vec = IndexedVector(Lambda_in, eye[i, :])
                res = applicator.apply(vec)
                resmat[i, :] = res.asarray()
                for j, mu in enumerate(sorted(Lambda_out)):
                    supp_mu = basis_out.wavelet_support(mu)
                    supp_total = supp_labda.intersection(supp_mu)
                    true_val = 0.0
                    if supp_total:
                        true_val = quad(lambda x: basis_in.eval_wavelet(
                            labda, x, deriv=deriv[0]) * basis_out.eval_wavelet(
                                mu, x, deriv=deriv[1]),
                                        supp_total.a,
                                        supp_total.b,
                                        points=[
                                            supp_labda.a, supp_labda.mid,
                                            supp_labda.b, supp_mu.a,
                                            supp_mu.mid, supp_mu.b,
                                            supp_total.a, supp_total.mid,
                                            supp_total.b
                                        ])[0]
                        truemat[i, j] = true_val
            try:
                assert np.allclose(resmat, truemat)
            except AssertionError:
                print(basis_in, basis_out)
                print(sorted(Lambda_in), sorted(Lambda_out))
                print(np.round(resmat, decimals=3))
                print(np.round(truemat, decimals=3))
                raise


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
            applicator = Applicator(basis, basis.singlescale_mass(), Lambda)
            for _ in range(10):
                c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
                res_full_op = applicator.apply(c)
                res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
                assert np.allclose(res_full_op.asarray(),
                                   res_upp_low.asarray())
