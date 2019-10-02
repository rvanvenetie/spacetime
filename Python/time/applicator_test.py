from fractions import Fraction

import numpy as np
from pytest import approx
from scipy.integrate import quad

from . import applicator, operators
from .haar_basis import HaarBasis
from .orthonormal_basis import OrthonormalBasis
from .sparse_vector import SparseVector
from .three_point_basis import ThreePointBasis

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
            applicator = Applicator_class(operators.mass(basis), basis)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = SparseVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                assert np.allclose(
                    [(1.0 if phi.labda[0] == 0 else 2**(phi.labda[0] - 1)) *
                     res[phi] for phi in Lambda_l], np_vec)


def test_orthonormal_multiscale_mass():
    """ Test that the multiscale mass operator is the identity. """
    for basis, Lambda in [
            OrthonormalBasis.uniform_basis(max_level=5),
            OrthonormalBasis.origin_refined_basis(max_level=15)
    ]:
        for l in range(1, 6):
            Lambda_l = Lambda.until_level(l)
            applicator = Applicator_class(operators.mass(basis), basis)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = SparseVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                for psi, val in vec.items():
                    assert val == approx(res[psi])


def test_multiscale_operator_quadrature_lin_comb():
    """ Test that the multiscale matrix equals that found with quadrature. """
    uml = 4
    oml = 11
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    hbe = HaarBasis.end_points_refined_basis(max_level=oml)
    oru = OrthonormalBasis.uniform_basis(max_level=uml)
    oro = OrthonormalBasis.origin_refined_basis(max_level=oml)
    ore = OrthonormalBasis.end_points_refined_basis(max_level=oml)
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
        print('\tLambda_in:\tdofs={}\tml={}'.format(len(Lambda_in.functions),
                                                    Lambda_in.maximum_level))
        print('\tLambda_out:\tdofs={}\tml={}'.format(len(Lambda_out.functions),
                                                     Lambda_out.maximum_level))
        applicator = Applicator_class(operator, basis_in, basis_out)
        for _ in range(3):
            vec_in = SparseVector(Lambda_in, np.random.rand(len(Lambda_in)))
            vec_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            applicator.apply(vec_in, vec_out)

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
                h = Fraction(1, 2**uml)
                while points[-1] < supp_psi_out[1]:
                    points.append(points[-1] + h)
                points = list(map(float, points))

                # Apply quadrature.
                true_val = quad(lambda x: psi_vec_in_eval(x, deriv=deriv[0]) *
                                psi_out.eval(x, deriv=deriv[1]),
                                supp_psi_out[0],
                                supp_psi_out[1],
                                points=points)[0]

                assert true_val == approx(vec_out[psi_out])


def test_apply_upp_low_vs_full():
    """ Test that apply_upp() + apply_low() == apply(). """
    uml = 6
    oml = 15
    hbu = HaarBasis.uniform_basis(max_level=uml)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    hbe = HaarBasis.end_points_refined_basis(max_level=oml)
    oru = OrthonormalBasis.uniform_basis(max_level=uml)
    oro = OrthonormalBasis.origin_refined_basis(max_level=oml)
    ore = OrthonormalBasis.end_points_refined_basis(max_level=oml)
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
        applicator = Applicator_class(operator, basis_in, basis_out)
        for _ in range(10):
            vec_in = SparseVector(Lambda_in, np.random.rand(len(Lambda_in)))
            res_full_op = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            vec_upp_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            vec_low_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))

            applicator.apply(vec_in, res_full_op)
            applicator.apply_upp(vec_in, vec_upp_out)
            applicator.apply_low(vec_in, vec_low_out)
            res_upp_low = vec_upp_out + vec_low_out
            assert np.allclose(res_full_op.asarray(Lambda_in),
                               res_upp_low.asarray(Lambda_in))
