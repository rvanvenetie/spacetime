from fractions import Fraction

import numpy as np
from pytest import approx

from ..datastructures.tree import NodeInterface
from .basis import MultiscaleFunctions, Scaling, Wavelet
from .haar_basis import HaarBasis
from .linear_operator_test import check_linop_transpose
from .orthonormal_basis import OrthonormalBasis
from .three_point_basis import ThreePointBasis


def test_haar_mother_functions():
    basis = HaarBasis()
    assert len(basis.metaroot_scaling.roots) == 1
    assert len(basis.metaroot_wavelet.roots) == 1
    mother_scaling = basis.metaroot_scaling.roots[0]

    assert mother_scaling.labda == (0, 0)
    assert mother_scaling.eval(0.25) == 1.0
    assert mother_scaling.eval(0.85) == 1.0

    basis.metaroot_wavelet.roots[0].refine()
    assert len(basis.metaroot_wavelet.roots[0].children) == 1
    mother_wavelet = basis.metaroot_wavelet.roots[0].children[0]

    assert mother_wavelet.labda == (1, 0)
    assert mother_wavelet.eval(0.25) == 1.0
    assert mother_wavelet.eval(0.85) == -1.0
    assert mother_wavelet.single_scale == [
        (basis.mother_scaling.children[0], 1),
        (basis.mother_scaling.children[1], -1)
    ]


def test_haar_uniform_refinement():
    ml = 5
    HaarBasis.metaroot_wavelet.uniform_refine(5)
    Lambda = MultiscaleFunctions(HaarBasis.metaroot_wavelet)
    Delta = MultiscaleFunctions(HaarBasis.metaroot_scaling)

    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**(l - 1)
        assert len(Delta.per_level[l]) == 2**l
        h = Fraction(1, 2**(l - 1))
        for psi in Lambda.per_level[l]:
            psi_l, psi_n = psi.labda
            assert psi_l == l
            assert psi.support[0].interval[0] == h * psi_n
            assert psi.support[-1].interval[1] == h * (psi_n + 1)

            assert psi.eval(h * psi_n + 0.25 * h) == 1.0
            assert psi.eval(h * psi_n + 0.75 * h) == -1.0


def test_haar_local_refinement():
    ml = 8
    basis, Lambda = HaarBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_functions()

    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 1
        assert len(Delta.per_level[l]) == 2
        assert Lambda.per_level[l][0].labda == (l, 0)
        #assert Delta.per_level[l][0].labda == (l, 0) and Delta.per_level[l][1].labda == (l, 1)


def test_ortho_uniform_refinement():
    sq3 = np.sqrt(3)
    ml = 5
    OrthonormalBasis.metaroot_wavelet.uniform_refine(ml)
    Lambda = MultiscaleFunctions(OrthonormalBasis.metaroot_wavelet)
    Delta = MultiscaleFunctions(OrthonormalBasis.metaroot_scaling)
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**l
        assert len(Delta.per_level[l]) == 2**(l + 1)
        h = Fraction(1, 2**l)
        h_supp = Fraction(1, 2**(l - 1))
        for psi in Lambda.per_level[l]:
            assert len(psi.parents) == 2
            assert len(psi.children) in [0, 4]
            for i in range(len(psi.children) - 1):
                assert psi.children[i].labda[1] + 1 == psi.children[i +
                                                                    1].labda[1]
            psi_l, psi_n = psi.labda
            assert psi_l == l
            assert psi.support[0].interval[0] == h_supp * (psi_n // 2)
            assert psi.support[-1].interval[1] == h_supp * (psi_n // 2 + 1)
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
    basis, Lambda = OrthonormalBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_functions()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2
        assert len(Delta.per_level[l]) == 4
        assert Lambda.per_level[l][0].labda == (l, 0)


def test_3pt_uniform_refinement():
    ml = 7
    ThreePointBasis.metaroot_wavelet.uniform_refine(ml)
    Lambda = MultiscaleFunctions(ThreePointBasis.metaroot_wavelet)
    Delta = MultiscaleFunctions(ThreePointBasis.metaroot_scaling)
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 2**(l - 1)
        assert len(Delta.per_level[l]) == 2**l + 1
        h = Fraction(1, 2**l)
        for psi in Lambda.per_level[l]:
            psi_l, psi_n = psi.labda
            assert psi_l == l
            assert psi.eval(h * (2 * psi_n + 1)) == 2**(l / 2)
            if psi_n > 0:
                assert psi.eval(h * (2 * psi_n)) == 0.5 * -2**(l / 2)
                assert psi.support[0].interval[0] == h * (2 * psi_n - 1)
            if psi_n < 2**(l - 1) - 1:
                assert psi.eval(h * (2 * psi_n + 2)) == 0.5 * -2**(l / 2)
                assert psi.support[-1].interval[1] == h * (2 * psi_n + 3)
            if psi_n == 0:
                assert psi.support[0].interval[0] == h * (2 * psi_n)
                assert psi.eval(0) == -2**(l / 2)
            if psi_n == 2**(l - 1) - 1:
                assert psi.support[-1].interval[1] == h * (2 * psi_n + 2)
                assert psi.eval(1) == -2**(l / 2)


def test_3pt_local_refinement():
    ml = 8
    basis, Lambda = ThreePointBasis.origin_refined_basis(ml)
    Delta = Lambda.single_scale_functions()
    for l in range(1, ml + 1):
        assert len(Lambda.per_level[l]) == 1
        assert len(Delta.per_level[l]) == 3
        assert Lambda.per_level[l][0].labda == (l, 0)

    basis, Lambda = ThreePointBasis.end_points_refined_basis(ml)
    Delta = Lambda.single_scale_functions()
    for l in range(2, ml + 1):
        assert len(Lambda.per_level[l]) == 2
        assert len(Delta.per_level[l]) <= 6
        assert {psi.labda
                for psi in Lambda.per_level[l]} == {(l, 0),
                                                    (l, 2**(l - 1) - 1)}


def test_all_subclasses():
    for basis, Lambda in [
            HaarBasis.uniform_basis(3),
            OrthonormalBasis.uniform_basis(3),
            ThreePointBasis.uniform_basis(3)
    ]:
        assert all(isinstance(psi, NodeInterface) for psi in Lambda)
        assert all(
            isinstance(elem, NodeInterface) for psi in Lambda
            for elem in psi.support)


def test_basis_PQ():
    """ Test if we recover the scaling functions by applying P or Q. """
    t = np.linspace(0, 1, 1025)
    uml = 6
    oml = 20
    for basis, Lambda in [
            HaarBasis.uniform_basis(uml),
            HaarBasis.origin_refined_basis(oml),
            HaarBasis.end_points_refined_basis(oml),
            OrthonormalBasis.uniform_basis(uml),
            OrthonormalBasis.origin_refined_basis(oml),
            OrthonormalBasis.end_points_refined_basis(oml),
            ThreePointBasis.uniform_basis(uml),
            ThreePointBasis.origin_refined_basis(oml),
            ThreePointBasis.end_points_refined_basis(oml),
    ]:
        Delta = Lambda.single_scale_functions()
        ml = Lambda.maximum_level
        print('Print testing PQ for {} with {} levels.'.format(
            basis.__class__.__name__, ml))
        for l in range(1, ml):
            assert all([
                isinstance(phi, Scaling)
                for phi in basis.P.range(Delta.per_level[l - 1])
            ])
            assert all([
                isinstance(phi, Scaling)
                for phi in basis.P.domain(Delta.per_level[l])
            ])
            assert set(basis.P.domain(Delta.per_level[l])).issubset(
                set(Delta.per_level[l - 1]))
            assert all([
                isinstance(psi, Wavelet)
                for psi in basis.Q.domain(Delta.per_level[l])
            ])
            assert all([
                isinstance(psi, Scaling)
                for psi in basis.Q.range(Lambda.per_level[l])
            ])

            for i, phi in enumerate(Delta.per_level[l - 1]):
                # Write phi_mu on lv l-1 as combination of scalings on lv l.
                vec = {phi: 1.0}
                res = basis.P.matvec(vec)
                inner = np.sum([phi.eval(t) * res[phi] for phi in res], axis=0)
                try:
                    assert np.allclose(inner, phi.eval(t))
                except AssertionError:
                    import matplotlib.pyplot as plt
                    plt.plot(t, inner, label=r'$(\Phi_{l-1}^T P_l)_\mu$')
                    plt.plot(t, phi.eval(t), label=r"$\phi_\mu$")
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
            HaarBasis.end_points_refined_basis(oml),
            OrthonormalBasis.uniform_basis(uml),
            OrthonormalBasis.origin_refined_basis(oml),
            OrthonormalBasis.end_points_refined_basis(oml),
            ThreePointBasis.uniform_basis(uml),
            ThreePointBasis.origin_refined_basis(oml),
            ThreePointBasis.end_points_refined_basis(oml),
    ]:
        Delta = Lambda.single_scale_functions()
        ml = Lambda.maximum_level
        for l in range(1, ml):
            print('test PQ on level {} for {}'.format(
                l, basis.__class__.__name__))
            check_linop_transpose(basis.P, set(Delta.per_level[l - 1]),
                                  set(Delta.per_level[l]))
            check_linop_transpose(basis.Q, set(Lambda.per_level[l]),
                                  set(Delta.per_level[l]))


def print_3point_functions():
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, 1025)
    for basis, Lambda in [
            HaarBasis.uniform_basis(max_level=4),
            ThreePointBasis.uniform_basis(max_level=4),
    ]:
        for l in range(Lambda.maximum_level):
            for psi in Lambda.on_level(l):
                plt.plot(t, psi.eval(t), label=psi.labda)
            plt.title("Wavelet functions on level {} for {}".format(
                l, basis.__class__.__name__))
            plt.legend()
            plt.show()


#        for l in range(basis.indices.maximum_level + 1):
#            for labda in basis.indices.on_level(l):
#                plt.plot(t, basis.eval_wavelet(labda, t), label=labda)
#            plt.legend()
#            plt.show()

if __name__ == "__main__":
    print_3point_functions()
