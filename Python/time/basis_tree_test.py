from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

from basis_tree import (BaseScaling, BaseWavelet, HaarBasis, OrthoBasis,
                        ThreePointBasis, sq3)
from indexed_vector import IndexedVector
from interval import Interval
from linear_operator_test import check_linop_transpose


def test_haar_mother_functions():
    basis = HaarBasis()

    assert basis.mother_scaling.labda == (0, 0)
    assert basis.mother_scaling.eval(0.25) == 1.0
    assert basis.mother_scaling.eval(0.85) == 1.0

    assert basis.mother_wavelet.labda == (1, 0)
    assert basis.mother_wavelet.eval(0.25) == 1.0
    assert basis.mother_wavelet.eval(0.85) == -1.0
    assert basis.mother_wavelet.single_scale == [
        (basis.mother_scaling.children[0], 1),
        (basis.mother_scaling.children[1], -1)
    ]


def test_haar_uniform_refinement():
    ml = 5
    basis, Lambda = HaarBasis.uniform_basis(ml)
    Delta = Lambda.single_scale_indices()

    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**(l - 1)
        assert len(Delta.per_level[l]) == 2**l
        h = Fraction(1, 2**(l - 1))
        for psi in Lambda.per_level[l]:
            psi_l, psi_n = psi.labda
            assert psi_l == l
            assert Interval(psi.support[0].interval.a,
                            psi.support[-1].interval.b) == Interval(
                                h * psi_n, h * (psi_n + 1))

            assert psi.eval(h * psi_n + 0.25 * h) == 1.0
            assert psi.eval(h * psi_n + 0.75 * h) == -1.0


def test_haar_local_refinement():
    ml = 8
    basis, Lambda = HaarBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_indices()

    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 1
        assert len(Delta.per_level[l]) == 2
        assert Lambda.per_level[l][0].labda == (l, 0)
        #assert Delta.per_level[l][0].labda == (l, 0) and Delta.per_level[l][1].labda == (l, 1)


def test_ortho_uniform_refinement():
    ml = 5
    basis, Lambda = OrthoBasis.uniform_basis(ml)
    Delta = Lambda.single_scale_indices()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**l
        assert len(Delta.per_level[l]) == 2**(l + 1)
        h = Fraction(1, 2**l)
        for psi in Lambda.per_level[l]:
            psi_l, psi_n = psi.labda
            assert psi_l == l
            if psi_n % 2 == 1:
                assert psi.eval(2 * h * (psi_n // 2) + 1e-8) == approx(
                    sq3 * 2**((l - 1) / 2))
                assert psi.eval(h + 2 * h * (psi_n // 2)) == approx(
                    -sq3 * 2**((l - 1) / 2))
                assert psi.eval(2 * h + 2 * h * (psi_n // 2) - 1e-8) == approx(
                    sq3 * 2**((l - 1) / 2))
            else:
                assert psi.eval(2 * h * (psi_n // 2) + 1e-8) == approx(
                    2**((l - 1) / 2))
                assert psi.eval(h + 2 * h * (psi_n // 2) - 1e-8) == approx(
                    -2 * 2**((l - 1) / 2))
                assert psi.eval(h + 2 * h * (psi_n // 2) + 1e-8) == approx(
                    2 * 2**((l - 1) / 2))
                assert psi.eval(2 * h + 2 * h * (psi_n // 2) -
                                1e-8) == approx(-2**((l - 1) / 2))


def test_ortho_local_refinement():
    ml = 8
    basis, Lambda = OrthoBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_indices()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2
        assert len(Delta.per_level[l]) == 4
        assert Lambda.per_level[l][0].labda == (l, 0)


def test_3pt_uniform_refinement():
    ml = 5
    basis, Lambda = ThreePointBasis.uniform_basis(ml)
    Delta = Lambda.single_scale_indices()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**(l - 1)
        assert len(Delta.per_level[l]) == 2**l + 1
        h = Fraction(1, 2**l)
        for psi in Lambda.per_level[l]:
            psi_l, psi_n = psi.labda
            assert psi_l == l
            assert psi.eval(h * (2 * psi_n + 1)) == 2**(l / 2)
            if psi_n > 0: assert psi.eval(h * (2 * psi_n)) == 0.5 * -2**(l / 2)
            if psi_n < 2**(l - 1) - 1:
                assert psi.eval(h * (2 * psi_n + 2)) == 0.5 * -2**(l / 2)
            if psi_n == 0: assert psi.eval(0) == -2**(l / 2)


def test_3pt_local_refinement():
    ml = 8
    basis, Lambda = ThreePointBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_indices()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 1
        assert len(Delta.per_level[l]) == 3
        assert Lambda.per_level[l][0].labda == (l, 0)

    basis, Lambda = ThreePointBasis.end_point_refined_basis(ml)
    Delta = Lambda.single_scale_indices()
    for l in range(2, ml + 1):
        assert len(Lambda.per_level[l]) == 2
        assert len(Delta.per_level[l]) <= 6
        assert {psi.labda
                for psi in Lambda.per_level[l]} == {(l, 0), (l,
                                                             2**(l - 1) - 1)}


def test_basis_PQ():
    """ Test if we recover the scaling functions by applying P or Q. """
    x = np.linspace(0, 1, 1025)
    uml = 6
    oml = 20
    for basis, Lambda in [
            HaarBasis.uniform_basis(uml),
            HaarBasis.origin_refined_basis(oml),
            HaarBasis.end_point_refined_basis(oml),
            OrthoBasis.uniform_basis(uml),
            OrthoBasis.origin_refined_basis(oml),
            OrthoBasis.end_point_refined_basis(oml),
            ThreePointBasis.uniform_basis(uml),
            ThreePointBasis.origin_refined_basis(oml),
            ThreePointBasis.end_point_refined_basis(oml),
    ]:
        Delta = Lambda.single_scale_indices()
        ml = Lambda.maximum_level
        print('Print testing PQ for {} with {} levels.'.format(
            basis.__class__.__name__, ml))
        for l in range(1, ml):
            assert all([
                isinstance(phi, BaseScaling)
                for phi in basis.P.range(Delta.per_level[l - 1])
            ])
            assert all([
                isinstance(phi, BaseScaling)
                for phi in basis.P.domain(Delta.per_level[l])
            ])
            assert set(basis.P.domain(Delta.per_level[l])).issubset(
                set(Delta.per_level[l - 1]))
            assert all([
                isinstance(psi, BaseWavelet)
                for psi in basis.Q.domain(Delta.per_level[l])
            ])
            assert all([
                isinstance(psi, BaseScaling)
                for psi in basis.Q.range(Lambda.per_level[l])
            ])

            eye = np.eye(len(Delta.per_level[l - 1]))
            for i, phi in enumerate(Delta.per_level[l - 1]):
                # Write phi_mu on lv l-1 as combination of scalings on lv l.
                vec = IndexedVector({phi: 1.0})
                res = basis.P.matvec(vec)
                inner = np.sum([phi.eval(x) * res[phi] for phi in res], axis=0)
                try:
                    assert np.allclose(inner, phi.eval(x))
                except AssertionError:
                    plt.plot(x, inner, label=r'$(\Phi_{l-1}^T P_l)_\mu$')
                    plt.plot(x, phi.eval(x), label=r"$\phi_\mu$")
                    plt.legend()
                    plt.show()
                    raise


def test_basis_PQ_matrix():
    """ Test that P.T = PT and Q.T = QT as matrices. """
    uml = 6
    oml = 20
    for basis, Lambda in [
            HaarBasis.uniform_basis(uml),
            HaarBasis.origin_refined_basis(oml),
            HaarBasis.end_point_refined_basis(oml),
            OrthoBasis.uniform_basis(uml),
            OrthoBasis.origin_refined_basis(oml),
            OrthoBasis.end_point_refined_basis(oml),
            ThreePointBasis.uniform_basis(uml),
            ThreePointBasis.origin_refined_basis(oml),
            ThreePointBasis.end_point_refined_basis(oml),
    ]:
        Delta = Lambda.single_scale_indices()
        ml = Lambda.maximum_level
        for l in range(1, ml):
            print('test PQ on level {} for {}'.format(
                l, basis.__class__.__name__))
            check_linop_transpose(basis.P, set(Delta.per_level[l - 1]),
                                  set(Delta.per_level[l]))
            check_linop_transpose(basis.Q, set(Lambda.per_level[l]),
                                  set(Delta.per_level[l]))


def test_haar_scaling_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    basis, Lambda = HaarBasis.uniform_basis(max_level=5)
    Delta = Lambda.single_scale_indices()
    mass = basis.scaling_mass()
    for l in range(1, 5):
        indices = Delta.on_level(l)
        assert len(indices) == 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**l))
            res = mass.matvec(d, set(indices), indices)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def print_3point_functions():
    x = np.linspace(0, 1, 1025)
    for basis, Lambda in [
            HaarBasis.uniform_basis(max_level=4),
            ThreePointBasis.uniform_basis(max_level=4),
    ]:
        for l in range(Lambda.maximum_level):
            for psi in Lambda.on_level(l):
                plt.plot(x, psi.eval(x), label=psi.labda)
            plt.title("Wavelet functions on level {} for {}".format(
                l, basis.__class__.__name__))
            plt.legend()
            plt.show()


#        for l in range(basis.indices.maximum_level + 1):
#            for labda in basis.indices.on_level(l):
#                plt.plot(x, basis.eval_wavelet(labda, x), label=labda)
#            plt.legend()
#            plt.show()

if __name__ == "__main__":
    print_3point_functions()
