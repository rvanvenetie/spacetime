from haar_basis import HaarBasis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis, ms2ss, ss2ms, position_ms

from index_set import MultiscaleIndexSet
from indexed_vector import IndexedVector

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pytest

np.set_printoptions(linewidth=10000, precision=3)


def test_haar_singlescale_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    basis = HaarBasis.uniform_basis(max_level=5)
    mass = basis.singlescale_mass
    for l in range(1, 5):
        indices = basis.scaling_indices_on_level(l)
        assert len(indices.indices) == 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**l))
            res = mass.matvec(indices, indices, d)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_orthonormal_singlescale_mass():
    basis = OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=4)
    mass = basis.singlescale_mass
    for l in range(1, 5):
        indices = basis.scaling_indices_on_level(l)
        assert len(indices.indices) == 2 * 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2 * 2**l))
            res = mass.matvec(indices, indices, d)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_basis_correct_support():
    N = 1024
    x = np.linspace(0, 1, N + 1)
    ml = 6
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            HaarBasis.origin_refined_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml),
            ThreePointBasis.origin_refined_basis(max_level=ml)
    ]:
        for l in range(1, ml + 1):
            Pi_B = basis.scaling_indices_on_level(l - 1)
            Pi_bar = basis.scaling_indices_on_level(l)
            Lambda_l = basis.indices.on_level(l)
            for i, mu in enumerate(Pi_B.asarray()):
                nz = np.nonzero(basis.eval_scaling(mu, x))[0]
                assert (nz[0] + 1) / N >= basis.scaling_support(
                    mu).a >= (nz[0] - 1) / N
                assert (nz[-1] - 1) / N <= basis.scaling_support(
                    mu).b <= (nz[-1] + 1) / N

            for i, mu in enumerate(Lambda_l.asarray()):
                nz = np.nonzero(basis.eval_wavelet(mu, x))[0]
                assert (nz[0] + 1) / N >= basis.wavelet_support(
                    mu).a >= (nz[0] - 1) / N
                assert (nz[-1] - 1) / N <= basis.wavelet_support(
                    mu).b <= (nz[-1] + 1) / N


def test_basis_PQ():
    """ Test if we recover the scaling functions by applying P or Q. """
    x = np.linspace(0, 1, 1025)
    ml = 6
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            HaarBasis.origin_refined_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml),
            ThreePointBasis.origin_refined_basis(max_level=ml)
    ]:
        for l in range(1, ml + 1):
            Pi_B = basis.scaling_indices_on_level(l - 1)
            Pi_bar = basis.scaling_indices_on_level(l)
            eye = np.eye(len(Pi_B))
            for i, mu in enumerate(Pi_B.asarray()):
                # Write phi_mu on lv l-1 as combination of scalings on lv l.
                vec = IndexedVector(Pi_B, eye[i, :])
                res = basis.P.matvec(Pi_B, Pi_bar, vec)
                inner = np.sum([
                    basis.eval_scaling(labda, x) * res[labda]
                    for labda in Pi_bar
                ],
                               axis=0)
                try:
                    assert np.allclose(inner, basis.eval_scaling(mu, x))
                except AssertionError:
                    plt.plot([
                        mu[1] / 2**l
                        for mu in basis.scaling_indices_on_level(l)
                    ], [0] * len(basis.scaling_indices_on_level(l)), 'ko')
                    plt.plot(x, inner, label=r'$(\Phi_{l-1}^T P_l)_\mu$')
                    plt.plot(x, basis.eval_scaling(mu, x), label=r"$\phi_\mu$")
                    plt.legend()
                    plt.show()
                    raise

            Lambda_l = basis.indices.on_level(l)
            for i, mu in enumerate(Lambda_l.asarray()):
                # Write psi_mu on lv l as combination of scalings on lv l.
                vec = IndexedVector(Lambda_l, eye[i, :])
                res = basis.Q.matvec(Lambda_l, Pi_bar, vec)
                inner = np.sum([
                    basis.eval_scaling(labda, x) * res[labda]
                    for labda in Pi_bar
                ],
                               axis=0)
                try:
                    assert np.allclose(inner, basis.eval_wavelet(mu, x))
                except AssertionError:
                    print(basis)
                    plt.plot([
                        position_ms(mu) for mu in basis.indices.until_level(l)
                    ], [0] * len(basis.indices.until_level(l)), 'ko')
                    plt.plot(x, inner, label=r'$(\Phi_{l-1}^T Q_l)_\mu$')
                    plt.plot(x, basis.eval_wavelet(mu, x), label=r"$\psi_\mu$")
                    plt.legend()
                    plt.show()
                    raise


def test_basis_PQ_matrix():
    """ Test that P.T = PT and Q.T = QT as matrices. """
    ml = 6
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            HaarBasis.origin_refined_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml),
            ThreePointBasis.origin_refined_basis(max_level=ml)
    ]:
        for l in range(1, ml + 1):
            Pi_B = basis.scaling_indices_on_level(l - 1)
            Pi_bar = basis.scaling_indices_on_level(l)
            Lambda_l = basis.indices.on_level(l)

            B_eye = np.eye(len(Pi_B))
            bar_eye = np.eye(len(Pi_bar))
            l_eye = np.eye(len(Lambda_l))

            P = np.zeros([len(Pi_B), len(Pi_bar)])
            PT = np.zeros([len(Pi_bar), len(Pi_B)])
            Q = np.zeros([len(Lambda_l), len(Pi_bar)])
            QT = np.zeros([len(Pi_bar), len(Lambda_l)])

            for i, mu in enumerate(Pi_B.asarray()):
                vec = IndexedVector(Pi_B, B_eye[i, :])
                P[i, :] = basis.P.matvec(Pi_B, Pi_bar, vec).asarray()
            for i, mu in enumerate(Pi_bar.asarray()):
                vec = IndexedVector(Pi_bar, bar_eye[i, :])
                PT[i, :] = basis.P.rmatvec(Pi_bar, Pi_B, vec).asarray()
            for i, mu in enumerate(Lambda_l.asarray()):
                vec = IndexedVector(Lambda_l, l_eye[i, :])
                Q[i, :] = basis.Q.matvec(Lambda_l, Pi_bar, vec).asarray()
            for i, mu in enumerate(Pi_bar.asarray()):
                vec = IndexedVector(Pi_bar, bar_eye[i, :])
                QT[i, :] = basis.Q.rmatvec(Pi_bar, Lambda_l, vec).asarray()

            print(Q.T, QT)
            assert np.allclose(P.T, PT)
            assert np.allclose(Q.T, QT)
            assert np.allclose(PT @ P, (PT @ P).T)
            assert np.allclose(QT @ Q, (QT @ Q).T)


def test_3point_ss2ms():
    assert ss2ms((0, 0)) == (0, 0)
    assert ss2ms((0, 1)) == (0, 1)

    assert ss2ms((1, 0)) == (0, 0)
    assert ss2ms((1, 1)) == (1, 0)
    assert ss2ms((1, 2)) == (0, 1)

    assert ss2ms((2, 0)) == (0, 0)
    assert ss2ms((2, 1)) == (2, 0)
    assert ss2ms((2, 2)) == (1, 0)
    assert ss2ms((2, 3)) == (2, 1)
    assert ss2ms((2, 4)) == (0, 1)

    assert ss2ms((3, 0)) == (0, 0)
    assert ss2ms((3, 1)) == (3, 0)
    assert ss2ms((3, 2)) == (2, 0)
    assert ss2ms((3, 3)) == (3, 1)
    assert ss2ms((3, 4)) == (1, 0)
    assert ss2ms((3, 5)) == (3, 2)
    assert ss2ms((3, 6)) == (2, 1)
    assert ss2ms((3, 7)) == (3, 3)
    assert ss2ms((3, 8)) == (0, 1)

    assert ss2ms((4, 0)) == (0, 0)
    assert ss2ms((4, 1)) == (4, 0)
    assert ss2ms((4, 2)) == (3, 0)
    assert ss2ms((4, 3)) == (4, 1)
    assert ss2ms((4, 4)) == (2, 0)
    assert ss2ms((4, 5)) == (4, 2)
    assert ss2ms((4, 6)) == (3, 1)
    assert ss2ms((4, 7)) == (4, 3)
    assert ss2ms((4, 8)) == (1, 0)
    assert ss2ms((4, 9)) == (4, 4)
    assert ss2ms((4, 10)) == (3, 2)
    assert ss2ms((4, 11)) == (4, 5)
    assert ss2ms((4, 12)) == (2, 1)
    assert ss2ms((4, 13)) == (4, 6)
    assert ss2ms((4, 14)) == (3, 3)
    assert ss2ms((4, 15)) == (4, 7)
    assert ss2ms((4, 16)) == (0, 1)

    for l in range(1, 10):
        for n in range(0, 2**l + 1):
            assert ms2ss(l, ss2ms((l, n))) == (l, n)


def test_3point_singlescale_indices():
    multiscale_indices = MultiscaleIndexSet({(0, 0), (0, 1), (1, 0), (2, 0),
                                             (2, 1), (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    assert basis.scaling_indices_on_level(1).indices == {(1, 0), (1, 1),
                                                         (1, 2)}
    assert basis.scaling_indices_on_level(2).indices == {(2, 0), (2, 1),
                                                         (2, 2), (2, 3),
                                                         (2, 4)}
    assert basis.scaling_indices_on_level(3).indices == {
        (3, 0), (3, 1), (3, 2), (3, 4), (3, 6), (3, 7), (3, 8)
    }


def test_singlescale_mass_quadrature():
    """ Use quadrature to test the singlescale mass matrices. """
    ml = 4
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            HaarBasis.origin_refined_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml),
            ThreePointBasis.origin_refined_basis(max_level=ml)
    ]:
        mass = basis.singlescale_mass
        for l in range(1, ml + 1):
            Delta_l = basis.scaling_indices_on_level(l)
            eye = np.eye(len(Delta_l))
            for i, labda in enumerate(Delta_l.asarray()):
                phi_supp = basis.scaling_support(labda)
                unit_vec = IndexedVector(Delta_l, eye[i, :])
                out = mass.matvec(Delta_l, Delta_l, unit_vec)
                for mu in Delta_l.asarray():
                    supp = phi_supp.intersection(basis.scaling_support(mu))
                    if supp:
                        assert np.isclose(
                            out[mu],
                            quad(
                                lambda x: basis.eval_scaling(labda, x) * basis.
                                eval_scaling(mu, x), supp.a, supp.b)[0])
                    else:
                        assert np.isclose(out[mu], 0.0)


def print_3point_functions():
    x = np.linspace(0, 1, 1025)
    for basis in [
            ThreePointBasis.uniform_basis(max_level=4),
            ThreePointBasis.origin_refined_basis(max_level=4)
    ]:
        for l in range(1, basis.indices.maximum_level):
            for labda in basis.scaling_indices_on_level(l):
                plt.plot(
                    [mu[1] / 2**l for mu in basis.scaling_indices_on_level(l)],
                    [0] * len(basis.scaling_indices_on_level(l)), 'ko')
                plt.plot(x, basis.eval_scaling(labda, x), label=labda)
            plt.legend()
            plt.show()

        for labda in basis.indices:
            plt.plot([position_ms(mu) for mu in basis.indices],
                     [0] * len(basis.indices), 'ko')
            plt.plot(x, basis.eval_wavelet(labda, x), label=labda)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print_3point_functions()
