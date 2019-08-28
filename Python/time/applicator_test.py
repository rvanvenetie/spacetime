import operators
from fractions import Fraction

import numpy as np
from pytest import approx
from scipy.integrate import quad

import applicator
from basis import HaarBasis, OrthoBasis, ThreePointBasis
from sparse_vector import SparseVector

np.random.seed(0)
np.set_printoptions(linewidth=10000, precision=3)

Applicator_class = applicator.Applicator


def support_to_interval(elems):
    return (elems[0].interval[0], elems[-1].interval[-1])


def test_haar_multiscale_mass():
    """ Test that the multiscale Haar mass matrix is indeed diagonal. """
    for basis, Lambda in [
            HaarBasis.uniform_basis(max_level=5),
            HaarBasis.origin_refined_basis(max_level=10)
    ]:
        for l in range(1, 6):
            Lambda_l = Lambda.until_level(l)
            applicator = Applicator_class(basis, operators.mass(basis),
                                          Lambda_l)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = SparseVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                assert np.allclose([(1.0 if phi.labda[0] == 0 else 2**
                                     (phi.labda[0] - 1)) * res[phi]
                                    for phi in Lambda_l], np_vec)


def test_orthonormal_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    for basis, Lambda in [
            OrthoBasis.uniform_basis(max_level=5),
            OrthoBasis.origin_refined_basis(max_level=15)
    ]:
        for l in range(1, 6):
            Lambda_l = Lambda.until_level(l)
            applicator = Applicator_class(basis, operators.mass(basis),
                                          Lambda_l)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = SparseVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                for psi, val in vec.items():
                    assert val == approx(res[psi])


def test_multiscale_mass_quadrature():
    """ Test that the multiscale matrix equals that found with quadrature. """
    uml = 5
    oml = 15
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    oru = OrthoBasis.uniform_basis(max_level=uml)
    oro = OrthoBasis.origin_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    deriv = (False, False)
    for basis_in, basis_out in [(hbu, hbu), (hbo, hbo), (hbu, hbo), (oru, oru),
                                (oro, oro), (oru, oro), (tpu, tpu), (tpo, tpo),
                                (tpu, tpo), (hbu, tpu), (tpo, hbu), (hbo,
                                                                     tpu)]:
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator_class(basis_in, operator, Lambda_in, basis_out,
                                      Lambda_out)
        eye = np.eye(len(Lambda_in))
        resmat = np.zeros([len(Lambda_in), len(Lambda_out)])
        truemat = np.zeros([len(Lambda_in), len(Lambda_out)])
        for i, psi in enumerate(Lambda_in):
            supp_psi = support_to_interval(psi.support)
            vec = SparseVector({psi: 1})
            res = applicator.apply(vec)
            resmat[i, :] = res.asarray(Lambda_out)
            for j, mu in enumerate(Lambda_out):
                supp_mu = support_to_interval(mu.support)
                true_val = 0.0
                true_val = quad(lambda x: psi.eval(x, deriv=deriv[0]) * mu.
                                eval(x, deriv=deriv[1]),
                                supp_psi[0],
                                supp_psi[1],
                                points=[
                                    float(supp_psi[0]),
                                    float(supp_psi[1]),
                                    float(supp_mu[0]),
                                    float(supp_mu[1]),
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


def test_multiscale_operator_quadrature_lin_comb():
    """ Test that the multiscale matrix equals that found with quadrature. """
    uml = 4
    oml = 5
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    hbe = HaarBasis.end_points_refined_basis(max_level=oml)
    oru = OrthoBasis.uniform_basis(max_level=uml)
    oro = OrthoBasis.origin_refined_basis(max_level=oml)
    ore = OrthoBasis.end_points_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    tpe = ThreePointBasis.end_points_refined_basis(max_level=oml)
    deriv = (False, False)
    for basis_in, basis_out in [(hbu, hbu), (hbo, hbo), (hbu, hbo), (hbu, hbe),
                                (oru, oru), (oro, oro), (oru, oro), (oru, ore),
                                (tpu, tpu), (tpo, tpo), (tpu, tpo), (tpu, tpe),
                                (tpo, tpe), (hbu, tpe), (tpu, hbe), (hbu, tpu),
                                (hbu, tpe), (tpo, hbe)]:
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator_class(basis_in, operator, Lambda_in, basis_out,
                                      Lambda_out)
        for _ in range(3):
            vec_in = SparseVector(Lambda_in, np.random.rand(len(Lambda_in)))
            vec_out = applicator.apply(vec_in)

            # Define function that evaluates lin. comb. Psi_Lambda_in
            def psi_vec_in_eval(x, deriv):
                result = 0
                for psi_in, val_in in vec_in.items():
                    result += val_in * psi_in.eval(x, deriv)
                return result

            for psi_out in Lambda_out:
                supp_psi_out = support_to_interval(psi_out.support)

                # Calculate all the breakpoints in this interval.
                points = [supp_psi_out[0]]
                h = Fraction(
                    1, 2**max(Lambda_in.maximum_level,
                              Lambda_out.maximum_level))
                while points[-1] < supp_psi_out[1]:
                    points.append(points[-1] + h)
                points = list(map(float, points))

                # Apply quadrature.
                true_val = quad(lambda x: psi_vec_in_eval(x, deriv=deriv[0]) * psi_out.eval(x, deriv=deriv[1]),
                                supp_psi_out[0],
                                supp_psi_out[1], points=points)[0]

                assert true_val == approx(vec_out[psi_out])


def test_apply_upp_low_vs_full():
    """ Test that apply_upp() + apply_low() == apply(). """
    uml = 6
    oml = 15
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    hbe = HaarBasis.end_points_refined_basis(max_level=oml)
    oru = OrthoBasis.uniform_basis(max_level=uml)
    oro = OrthoBasis.origin_refined_basis(max_level=oml)
    ore = OrthoBasis.end_points_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    tpe = ThreePointBasis.end_points_refined_basis(max_level=oml)
    deriv = (False, False)
    for basis_in, basis_out in [(hbu, hbu), (hbo, hbo), (hbu, hbo), (hbu, hbe),
                                (oru, oru), (oro, oro), (oru, oro), (oru, ore),
                                (tpu, tpu), (tpo, tpo), (tpu, tpo), (tpu, tpe),
                                (tpo, tpe), (hbu, tpe), (tpu, hbe), (hbu, tpu),
                                (hbu, tpe), (tpo, hbe)]:
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        applicator = Applicator_class(basis_in, operator, Lambda_in, basis_out,
                                      Lambda_out)
        for _ in range(10):
            c = SparseVector(Lambda_in, np.random.rand(len(Lambda_in)))
            res_full_op = applicator.apply(c)
            res_upp_low = applicator.apply_upp(c) + applicator.apply_low(c)
            assert np.allclose(
                res_full_op.asarray(Lambda_in), res_upp_low.asarray(Lambda_in))
