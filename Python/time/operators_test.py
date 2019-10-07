from collections import defaultdict
from itertools import product

import numpy as np
from pytest import approx

from . import operators
from .haar_basis import HaarBasis
from .linear_operator_test import check_linop_transpose
from .orthonormal_basis import OrthonormalBasis
from .sparse_vector import SparseVector
from .three_point_basis import ThreePointBasis


def test_haar_scaling_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    basis, Lambda = HaarBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_functions()
    mass = operators.mass(basis)
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**l

        for _ in range(100):
            d = SparseVector(indices, np.random.rand(2**l))
            res = mass.matvec(d, set(indices), indices)
            assert np.allclose(d.asarray(indices),
                               2.0**l * res.asarray(indices))


def test_ortho_scaling_mass():
    """ Test that the ortho scaling mass matrix is indeed diagonal. """
    basis, Lambda = OrthonormalBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_functions()
    mass = operators.mass(basis)
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**(l + 1)

        for _ in range(100):
            d = SparseVector(indices, np.random.rand(2**(l + 1)))
            res = mass.matvec(d, set(indices), indices)
            assert np.allclose(d.asarray(indices),
                               2.0**l * res.asarray(indices))


def test_three_point_scaling_mass():
    """ Test the three point_scaling mass. """
    basis, Lambda = ThreePointBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_functions()
    mass = operators.mass(basis)
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**l + 1

        for phi in indices:
            _, n = phi.labda
            row = mass.row(phi)
            if n == 0 or n == 2**l:
                assert len(row) == 2
                assert (phi, 1 / 3 * 2**-l) in row
            else:
                assert len(row) == 3
                assert (phi, 2 / 3 * 2**-l) in row


def test_haar_three_scaling_mass():
    uml = 5
    basis_haar, Lambda_haar = HaarBasis.uniform_basis(uml)
    Delta_haar = Lambda_haar.single_scale_functions()

    basis_three, Lambda_three = ThreePointBasis.uniform_basis(uml)
    Delta_three = Lambda_three.single_scale_functions()

    mass_haar_three = operators.mass(basis_haar, basis_three)
    mass_three_haar = operators.mass(basis_three, basis_haar)

    for l in range(1, uml):
        indices_haar = Delta_haar.per_level[l]
        indices_three = Delta_three.per_level[l]
        check_linop_transpose(mass_haar_three, set(indices_haar),
                              set(indices_three))
        check_linop_transpose(mass_three_haar, set(indices_three),
                              set(indices_haar))
        for _ in range(10):
            d = SparseVector(indices_haar, np.random.rand(2**l))
            res = mass_haar_three.matvec(d, set(indices_haar),
                                         set(indices_three))
            res_test = defaultdict(float)
            for phi_haar, val in d.items():
                for phi_three in phi_haar.support[0].phi_cont_lin:
                    res_test[phi_three] += val * 1 / 2 * 2**-l
            for phi_three, val in res.items():
                assert res_test[phi_three] == approx(val)


def test_threepoint_trace():
    oru = OrthonormalBasis.uniform_basis(max_level=5)
    oro = OrthonormalBasis.origin_refined_basis(max_level=12)
    tpu = ThreePointBasis.uniform_basis(max_level=5)
    tpo = ThreePointBasis.origin_refined_basis(max_level=12)
    for (b_in, L_in), (b_out, L_out) in list(product([tpu, tpo], [tpu, tpo])) \
                                      + list(product([tpu, tpo], [oru, oro])) \
                                      + list(product([oru, oro], [tpu, tpo])):
        print('Calculating results for: basis_in={}\tbasis_out={}'.format(
            b_in.__class__.__name__, b_out.__class__.__name__))
        print('\tLambda_in:\tdofs={}\tml={}'.format(len(L_in.functions),
                                                    L_in.maximum_level))
        print('\tLambda_out:\tdofs={}\tml={}'.format(len(L_out.functions),
                                                     L_out.maximum_level))
        Delta_in = L_in.single_scale_functions()
        Delta_out = L_out.single_scale_functions()
        trace = operators.trace(b_in, b_out)
        for l in range(min(L_in.maximum_level, L_out.maximum_level) + 1):
            ind_in = Delta_in.on_level(l)
            ind_out = Delta_out.on_level(l)
            check_linop_transpose(trace, set(ind_in), set(ind_out))
            for _ in range(3):
                vec_in = SparseVector(ind_in, np.random.rand(len(ind_in)))
                vec_out = trace.matvec(vec_in, set(ind_in), set(ind_out))
                for psi_out in ind_out:
                    expected_out = sum(vec_in[psi_in] * psi_in.eval(0) *
                                       psi_out.eval(0) for psi_in in ind_in)
                    assert np.isclose(vec_out[psi_out], expected_out)
