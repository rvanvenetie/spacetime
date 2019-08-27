from collections import defaultdict

import numpy as np
from pytest import approx

import operator_tree
from basis_tree import HaarBasis, OrthoBasis, ThreePointBasis
from indexed_vector import IndexedVector
from linear_operator_test import check_linop_transpose


def test_haar_scaling_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    basis, Lambda = HaarBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_indices()
    mass = operator_tree.mass(basis)
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**l))
            res = mass.matvec(d, set(indices), indices)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_ortho_scaling_mass():
    """ Test that the ortho scaling mass matrix is indeed diagonal. """
    basis, Lambda = OrthoBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_indices()
    mass = operator_tree.mass(basis)
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**(l + 1)

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**(l + 1)))
            res = mass.matvec(d, set(indices), indices)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_three_point_scaling_mass():
    """ Test the three point_scaling mass. """
    basis, Lambda = ThreePointBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_indices()
    mass = operator_tree.mass(basis)
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
    Delta_haar = Lambda_haar.single_scale_indices()

    basis_three, Lambda_three = ThreePointBasis.uniform_basis(uml)
    Delta_three = Lambda_three.single_scale_indices()

    mass_haar_three = operator_tree.mass(basis_haar, basis_three)
    mass_three_haar = operator_tree.mass(basis_three, basis_haar)

    for l in range(1, uml):
        indices_haar = Delta_haar.per_level[l]
        indices_three = Delta_three.per_level[l]
        check_linop_transpose(mass_haar_three, set(indices_haar),
                              set(indices_three))
        check_linop_transpose(mass_three_haar, set(indices_three),
                              set(indices_haar))
        for _ in range(10):
            d = IndexedVector(indices_haar, np.random.rand(2**l))
            res = mass_haar_three.matvec(d, set(indices_haar),
                                         set(indices_three))
            res_test = defaultdict(float)
            for phi_haar, val in d.items():
                for phi_three in phi_haar.support[0].phi_cont_lin:
                    res_test[phi_three] += val * 1 / 2 * 2**-l
            for phi_three, val in res.items():
                assert res_test[phi_three] == approx(val)
