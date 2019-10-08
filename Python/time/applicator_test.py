import itertools
from fractions import Fraction

import numpy as np
from pytest import approx
from scipy.integrate import quad

from . import operators
from ..datastructures.tree_view import NodeViewInterface
from .applicator import Applicator
from .basis import MultiscaleFunctions
from .haar_basis import HaarBasis
from .orthonormal_basis import OrthonormalBasis
from .sparse_vector import SparseVector
from .three_point_basis import ThreePointBasis

np.random.seed(0)
np.set_printoptions(linewidth=10000, precision=3)


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
            applicator = Applicator(operators.mass(basis), basis)
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
            applicator = Applicator(operators.mass(basis), basis)
            for _ in range(10):
                np_vec = np.random.rand(len(Lambda_l))
                vec = SparseVector(Lambda_l, np_vec)
                res = applicator.apply(vec)
                for psi, val in vec.items():
                    assert val == approx(res[psi])


def test_haar_3pt_mass():
    basis_in = HaarBasis()
    basis_out = ThreePointBasis()

    basis_in.metaroot_wavelet.uniform_refine(3)
    basis_out.metaroot_wavelet.uniform_refine(4)

    # Get a subset of all wavelets.
    Lambda_in = MultiscaleFunctions(
        [psi for psi in basis_in.metaroot_wavelet.bfs() if psi.level <= 2])
    Lambda_out = MultiscaleFunctions(
        [psi for psi in basis_out.metaroot_wavelet.bfs() if psi.level <= 3])

    applicator = Applicator(operators.mass(basis_in, basis_out), basis_in,
                            basis_out)
    real_mat = applicator.to_matrix(Lambda_in, Lambda_out)
    for _ in range(2):
        vec_in = SparseVector(Lambda_in, np.zeros(len(Lambda_in)))
        vec_in[Lambda_in.functions[0]] = 1.0
        vec_in[Lambda_in.functions[1]] = 0
        vec_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
        applicator.apply_low(vec_in, vec_out)
        assert vec_out[Lambda_out.functions[0]] == 0
        assert vec_out[Lambda_out.functions[1]] == 0
        applicator.apply_upp(vec_in, vec_out)
        assert vec_out[Lambda_out.functions[0]] == 0.5
        assert vec_out[Lambda_out.functions[1]] == 0.5


def test_multiscale_mass_quadrature():
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
    for basis_in, basis_out in itertools.product(
        [hbu, hbo, hbe, oru, oro, ore, tpu, tpo, tpe], repeat=2):
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator(operator, basis_in, basis_out)
        resmat = applicator.to_matrix(Lambda_in, Lambda_out)
        truemat = np.zeros([len(Lambda_out), len(Lambda_in)])
        for j, psi in enumerate(Lambda_in):
            supp_psi = support_to_interval(psi.support)
            for i, mu in enumerate(Lambda_out):
                supp_mu = support_to_interval(mu.support)
                true_val = quad(
                    lambda x: psi.eval(x) * mu.eval(x),
                    max(supp_mu[0], supp_psi[0]),
                    min(supp_mu[1], supp_psi[1]),
                )[0]
                truemat[i, j] = true_val
        assert np.allclose(resmat, truemat)


def test_multiscale_transport_quadrature():
    """ Test that the multiscale matrix equals that found with quadrature. """
    uml = 4
    oml = 11
    oru = OrthonormalBasis.uniform_basis(max_level=uml)
    oro = OrthonormalBasis.origin_refined_basis(max_level=oml)
    ore = OrthonormalBasis.end_points_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    tpe = ThreePointBasis.end_points_refined_basis(max_level=oml)
    for basis_in, basis_out in itertools.product(
        [oru, oro, ore, tpu, tpo, tpe], [tpu, tpo, tpe]):
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.transport(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator(operator, basis_in, basis_out)
        resmat = applicator_to_matrix(applicator, Lambda_in, Lambda_out)
        truemat = np.zeros([len(Lambda_out), len(Lambda_in)])
        for j, psi_in in enumerate(Lambda_in):
            supp_in = support_to_interval(psi_in.support)
            for i, psi_out in enumerate(Lambda_out):
                supp_out = support_to_interval(psi_out.support)
                true_val = quad(
                    lambda x: psi_in.eval(x) * psi_out.eval(x, deriv=True),
                    max(supp_in[0], supp_out[0]),
                    min(supp_in[1], supp_out[1]),
                )[0]
                assert np.allclose(resmat[i, j], true_val)


def test_multiscale_trace():
    """ Test that the multiscale matrix equals that found with evaluation. """
    oru = OrthonormalBasis.uniform_basis(max_level=5)
    oro = OrthonormalBasis.origin_refined_basis(max_level=12)
    tpu = ThreePointBasis.uniform_basis(max_level=5)
    tpo = ThreePointBasis.origin_refined_basis(max_level=12)
    for (basis_in, Lambda_in), (basis_out, Lambda_out) in \
            list(itertools.product([tpu, tpo], [tpu, tpo])) \
          + list(itertools.product([tpu, tpo], [oru, oro])) \
          + list(itertools.product([oru, oro], [tpu, tpo])):
        operator = operators.trace(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        applicator = Applicator(operator, basis_in, basis_out)
        resmat = applicator.to_matrix(Lambda_in, Lambda_out)
        truemat = np.zeros([len(Lambda_out), len(Lambda_in)])
        for j, psi in enumerate(Lambda_in):
            for i, mu in enumerate(Lambda_out):
                truemat[i, j] = psi.eval(0) * mu.eval(0)
        assert np.allclose(resmat, truemat)


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
                                (hbu, tpe), (tpo, hbe), (tpe, ore),
                                (ore, tpo)]:
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            basis_in.__class__.__name__, basis_out.__class__.__name__))
        print('\tLambda_in:\tdofs={}\tml={}'.format(len(Lambda_in.functions),
                                                    Lambda_in.maximum_level))
        print('\tLambda_out:\tdofs={}\tml={}'.format(len(Lambda_out.functions),
                                                     Lambda_out.maximum_level))
        applicator = Applicator(operator, basis_in, basis_out)
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

                assert np.allclose(true_val, vec_out[psi_out], rtol=1e-3)


def test_apply_upp_low_vs_full():
    """ Test that apply_upp() + apply_low() == apply(). """
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
                                (hbu, tpe), (tpo, hbe), (tpe, ore),
                                (ore, tpo)]:
        basis_in, Lambda_in = basis_in
        basis_out, Lambda_out = basis_out
        operator = operators.mass(basis_in, basis_out)
        applicator = Applicator(operator, basis_in, basis_out)
        for _ in range(10):
            vec_in = SparseVector(Lambda_in, np.random.rand(len(Lambda_in)))
            res_full_op = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            vec_upp_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            vec_low_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))

            applicator.apply_low(vec_in, vec_low_out)
            applicator.apply_upp(vec_in, vec_upp_out)
            applicator.apply(vec_in, res_full_op)
            res_upp_low = vec_upp_out + vec_low_out
            assert np.allclose(res_full_op.asarray(Lambda_in),
                               res_upp_low.asarray(Lambda_in))
