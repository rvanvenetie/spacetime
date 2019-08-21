from applicator import Applicator

from haar_basis import HaarBasis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis
from indexed_vector import IndexedVector
from index_set import MultiscaleIndexSet, SingleLevelIndexSet

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pytest

np.random.seed(0)
np.set_printoptions(linewidth=10000, precision=3)


def test_haar_largest_subset():
    basis = HaarBasis.uniform_basis(max_level=5)
    applicator = Applicator(basis, basis.scaling_mass())

    # Check the largest subset contained in mother scaling/wavelt
    assert applicator._largest_subset(5, basis, Pi_B=[(0,0)], Lambda_l = []) == basis.scaling_indices_on_level(5)
    assert applicator._largest_subset(5, basis, Pi_B=[], Lambda_l =[(0,0)]) == basis.scaling_indices_on_level(5)

    # Check for a singlescale function on level 4
    for index in range(2**4):
        assert applicator._largest_subset(5, basis, Pi_B=[(4,index)], Lambda_l = []) == SingleLevelIndexSet([(5,index*2), (5,index*2+1)])

    assert applicator._largest_subset(5, basis, Pi_B=[(4,0), (4,2), (4,3)], Lambda_l = []) == SingleLevelIndexSet([
        (5,0), (5,1), (5,4), (5,5), (5,6), (5,7)])


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    for basis in [
            HaarBasis.uniform_basis(max_level=5),
            HaarBasis.origin_refined_basis(max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.scaling_mass(), Lambda)
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
#            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
#                max_level=5)
    ]:
        for l in range(1, 6):
            Lambda = basis.indices.until_level(l)
            applicator = Applicator(basis, basis.scaling_mass(), Lambda)
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
    ml = 6
    hbu = HaarBasis.uniform_basis(max_level=ml)
    hbo = HaarBasis.origin_refined_basis(max_level=ml)
    oru = OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml)
    oro = OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
        max_level=ml)
    tpu = ThreePointBasis.uniform_basis(max_level=ml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=ml)
    for basis_in, basis_out, operator, deriv in [
        (hbu, hbu, hbu.scaling_mass(), (False, False)),
        (hbo, hbo, hbo.scaling_mass(), (False, False)),
        (hbu, hbo, hbu.scaling_mass(), (False, False)),
        (oru, oru, oru.scaling_mass(), (False, False)),
        (oro, oro, oro.scaling_mass(), (False, False)),
        (oru, oro, oru.scaling_mass(), (False, False)),
        (oru, oru, oru.scaling_damping(), (True, False)),
        (oro, oro, oro.scaling_damping(), (True, False)),
        (oru, oro, oru.scaling_damping(), (True, False)),
        (tpu, tpu, tpu.scaling_mass(), (False, False)),
        (tpo, tpo, tpo.scaling_mass(), (False, False)),
            # 3-point mass currently only works for the same in- and out basis.
            #(tpu, tpo, tpu.scaling_mass(tpo), (False, False)),
        (tpu, tpu, tpu.scaling_damping(), (True, False)),
        (tpo, tpo, tpo.scaling_damping(), (True, False)),
            #(tpu, tpo, tpu.singlescale_damping(tpo), (True, False)),
    ]:
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(basis_in.__class__.__name__, basis_out.__class__.__name__))
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
                                            float(supp_labda.a), float(supp_labda.mid),
                                            float(supp_labda.b), float(supp_mu.a),
                                            float(supp_mu.mid ), float(supp_mu.b),
                                            float(supp_total.a), float(supp_total.mid),
                                            float(supp_total.b)
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
            applicator = Applicator(basis, basis.scaling_mass(), Lambda)
            for _ in range(10):
                c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
                res_full_op = applicator.apply(c)
                res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
                assert np.allclose(res_full_op.asarray(),
                                   res_upp_low.asarray())
