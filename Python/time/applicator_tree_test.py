from applicator_tree import Applicator
from basis_tree import support_to_interval
from basis_tree import HaarBasis, ThreePointBasis, OrthoBasis
from indexed_vector import IndexedVector

import numpy as np
from scipy.integrate import quad

np.random.seed(0)
np.set_printoptions(linewidth=10000, precision=3)


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    for basis, Lambda, Delta in [
            HaarBasis.uniform_basis(max_level=5),
            #HaarBasis.origin_refined_basis(max_level=5)
    ]:
        for l in range(1, 6):
            Lambda_l = Lambda.until_level(l)
            applicator = Applicator(basis, basis.scaling_mass(), Lambda_l)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = IndexedVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                assert np.allclose(
                    [(1.0 if phi.labda[0] == 0 else 2**(phi.labda[0] - 1)) *
                     res[phi] for phi in Lambda_l], np_vec)


def test_orthonormal_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    for basis, Lambda, Delta in [
            OrthoBasis.uniform_basis(max_level=5),
            OrthoBasis.origin_refined_basis(max_level=15)
    ]:
        for l in range(1, 6):
            Lambda_l = Lambda.until_level(l)
            applicator = Applicator(basis, basis.scaling_mass(), Lambda_l)
            eye = np.eye(len(Lambda_l))
            res_matrix = np.zeros([len(Lambda_l), len(Lambda_l)])
            for i, psi in enumerate(Lambda_l):
                vec = IndexedVector({psi: 1.0})
                res = applicator.apply(vec)
                print(psi, res)
                res_matrix[:, i] = res.asarray(Lambda_l)
            assert np.allclose(eye, res_matrix)

            for _ in range(10):
                vec = np.random.rand(len(Lambda_l))
                res = applicator.apply(IndexedVector(Lambda_l, vec))
                assert np.allclose(vec, res.asarray(Lambda_l))


def test_multiscale_operator_quadrature():
    """ Test that the multiscale matrix equals that found with quadrature. """
    uml = 5
    oml = 15
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    oru = OrthoBasis.uniform_basis(max_level=uml)
    oro = OrthoBasis.origin_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    for basis_in, basis_out, operator, deriv in [
        (hbu, hbu, HaarBasis.scaling_mass(), (False, False)),
        (hbo, hbo, HaarBasis.scaling_mass(), (False, False)),
        (hbu, hbo, HaarBasis.scaling_mass(), (False, False)),
        (oru, oru, OrthoBasis.scaling_mass(), (False, False)),
        (oro, oro, OrthoBasis.scaling_mass(), (False, False)),
        (oru, oro, OrthoBasis.scaling_mass(), (False, False)),
        (tpu, tpu, ThreePointBasis.scaling_mass(), (False, False)),
        (tpo, tpo, ThreePointBasis.scaling_mass(), (False, False)),
    ]:
        basis_in, Lambda_in, Delta_in = basis_in
        basis_out, Lambda_out, Delta_out = basis_out
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator(basis_in, operator, Lambda_in, basis_out,
                                Lambda_out)
        eye = np.eye(len(Lambda_in))
        resmat = np.zeros([len(Lambda_in), len(Lambda_out)])
        truemat = np.zeros([len(Lambda_in), len(Lambda_out)])
        for i, psi in enumerate(Lambda_in):
            supp_psi = support_to_interval(psi.support)
            vec = IndexedVector(Lambda_in, eye[i, :])
            res = applicator.apply(vec)
            resmat[i, :] = res.asarray(Lambda_out)
            for j, mu in enumerate(Lambda_out):
                supp_mu = support_to_interval(mu.support)
                supp_total = supp_psi.intersection(supp_mu)
                true_val = 0.0
                if supp_total:
                    true_val = quad(lambda x: psi.eval(x, deriv=deriv[0]) * mu.
                                    eval(x, deriv=deriv[1]),
                                    supp_total.a,
                                    supp_total.b,
                                    points=[
                                        float(supp_psi.a),
                                        float(supp_psi.mid),
                                        float(supp_psi.b),
                                        float(supp_mu.a),
                                        float(supp_mu.mid),
                                        float(supp_mu.b),
                                        float(supp_total.a),
                                        float(supp_total.mid),
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
    uml = 5
    oml = 15
    for basis in [
            HaarBasis.uniform_basis(max_level=uml),
            HaarBasis.origin_refined_basis(max_level=oml),
            OrthoBasis.uniform_basis(max_level=uml),
            OrthoBasis.origin_refined_basis(max_level=oml),
            ThreePointBasis.uniform_basis(max_level=uml),
            ThreePointBasis.origin_refined_basis(max_level=oml)
    ]:
        basis, Lambda, Delta = basis
        applicator = Applicator(basis, basis.scaling_mass(), Lambda)
        for _ in range(10):
            c = IndexedVector(Lambda, np.random.rand(len(Lambda)))
            res_full_op = applicator.apply(c)
            res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
            assert np.allclose(res_full_op.asarray(Lambda),
                               res_upp_low.asarray(Lambda))
