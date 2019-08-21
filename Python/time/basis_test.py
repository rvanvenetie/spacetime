from haar_basis import HaarBasis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis #, ms2ss, ss2ms, position_ms

from index_set import MultiscaleIndexSet
from indexed_vector import IndexedVector
from linear_operator_test import check_linop_transpose

import numpy as np
from scipy.integrate import quad
from fractions import Fraction
import matplotlib.pyplot as plt
import pytest


np.set_printoptions(linewidth=10000, precision=3)

def F(x,y): return Fraction(x,y)


def test_haar_ss_nonzero_in_nbrhood():
    """ Test that we always return the correct non-zero function. """
    basis = HaarBasis([])

    eps = F(1,1000)

    # Check level 0
    assert basis.scaling_indices_nonzero_in_nbrhood(0, 1-eps).indices == {(0,0)}
    assert basis.scaling_indices_nonzero_in_nbrhood(0, 0).indices == {(0,0)}

    # Check level 2
    assert basis.scaling_indices_nonzero_in_nbrhood(2, 0).indices == {(2,0)}
    assert basis.scaling_indices_nonzero_in_nbrhood(2, F(1,4)-eps).indices == {(2,0)}
    assert basis.scaling_indices_nonzero_in_nbrhood(2, F(1,4)).indices == {(2,0), (2,1)}
    assert basis.scaling_indices_nonzero_in_nbrhood(2, F(1,2)).indices == {(2,1), (2,2)}
    assert basis.scaling_indices_nonzero_in_nbrhood(2, F(3,4)).indices == {(2,2), (2,3)}

    # Check level 40
    assert basis.scaling_indices_nonzero_in_nbrhood(40, 0).indices == {(40,0)}
    assert basis.scaling_indices_nonzero_in_nbrhood(40, F(13,2**40)).indices == {(40,12), (40,13)}
    assert basis.scaling_indices_nonzero_in_nbrhood(40, F(876,2**40)).indices == {(40,875), (40, 876)}

    # Check 500 points for levels 0..123.
    for l in range(123):
        for pt in range(max(1, 2**l - 500), 2**l):
            assert basis.scaling_indices_nonzero_in_nbrhood(l, pt * F(1, 2**l)).indices == {(l, pt-1),(l, pt)}

def test_orthonormal_ss_nonzero_in_nbrhood():
    """ Test that we always return the correct non-zero function. """
    basis = OrthonormalDiscontinuousLinearBasis([])

    for l in range(14):
        for n in range(2**(l+1)):
            labda = (l, n)
            support = basis.scaling_support(labda)

            print(labda, support)
            assert labda in basis.scaling_indices_nonzero_in_nbrhood(l, support.a)
            assert labda in basis.scaling_indices_nonzero_in_nbrhood(l, support.mid)
            assert labda in basis.scaling_indices_nonzero_in_nbrhood(l, support.b)

            assert len(basis.scaling_indices_nonzero_in_nbrhood(l, support.mid)) == 2

            if n > 1 and n < 2**(l+1)-2:
                assert len(basis.scaling_indices_nonzero_in_nbrhood(l, support.a)) == len(basis.scaling_indices_nonzero_in_nbrhood(l, support.b)) == 4


def test_haar_scaling_mass():
    """ Test that the singlescale Haar mass matrix is indeed diagonal. """
    basis = HaarBasis.uniform_basis(max_level=5)
    mass = basis.scaling_mass()
    for l in range(1, 5):
        indices = basis.scaling_indices_on_level(l)
        assert len(indices.indices) == 2**l

        for _ in range(100):
            d = IndexedVector(indices, np.random.rand(2**l))
            res = mass.matvec(indices, indices, d)
            assert np.allclose(d.asarray(), 2.0**l * res.asarray())


def test_orthonormal_scaling_mass():
    basis = OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=4)
    mass = basis.scaling_mass()
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
                ss_support = basis.scaling_support(mu)
                nz = np.nonzero(basis.eval_scaling(mu, x))[0]
                assert (nz[0] + 1) / N >= ss_support.a >= (nz[0] - 1) / N
                assert (nz[-1] - 1) / N <= ss_support.b <= (nz[-1] + 1) / N

            for i, mu in enumerate(Lambda_l.asarray()):
                ms_support = basis.wavelet_support(mu)
                nz = np.nonzero(basis.eval_wavelet(mu, x))[0]
                assert nz.shape[0]
                assert (nz[0] + 1) / N >= ms_support.a >= (nz[0] - 1) / N
                assert (nz[-1] - 1) / N <= ms_support.b <=  (nz[-1] + 1) / N


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
                inner = np.sum([ basis.eval_scaling(labda, x) * res[labda] for labda in Pi_bar ], axis=0)
                try:
                    assert np.allclose(inner, basis.eval_scaling(mu, x))
                except AssertionError:
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


def test_basis_PQ_domain_range():
    """ Test that the domains/ranges of operators P and Q are correct. """
    ml = 6
    for basis in [
            HaarBasis.uniform_basis(max_level=ml),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=ml),
            ThreePointBasis.uniform_basis(max_level=ml)]:
        for l in range(1, ml + 1):
            print('test PQ domain/range on level {} for {}'.format(l, basis.__class__.__name__))
            Delta_l = basis.scaling_indices_on_level(l)
            Lambda_l = basis.wavelet_indices_on_level(l)

            assert basis.P.range(Delta_l) == basis.scaling_indices_on_level(l+1)
            assert basis.P.domain(Delta_l) == basis.scaling_indices_on_level(l-1)

            assert basis.Q.range(Lambda_l) == Delta_l
            assert basis.Q.domain(Delta_l) == Lambda_l


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
            print('test PQ on level {} for {}'.format(l, basis.__class__.__name__))
            Pi_B = basis.scaling_indices_on_level(l - 1)
            Pi_bar = basis.scaling_indices_on_level(l)
            Lambda_l = basis.indices.on_level(l)

            check_linop_transpose(basis.P, Pi_B, Pi_bar)
            check_linop_transpose(basis.Q, Lambda_l, Pi_bar)



def test_singlescale_quadrature():
    """ Use quadrature to test the singlescale matrices. """
    ml = 4
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
            #(tpu, oru, tpu.singlescale_damping(oru), (True, False)),
            #(tpo, oro, tpo.singlescale_damping(oro), (True, False)),
            #(tpu, oro, tpu.singlescale_damping(oro), (True, False)),
        (tpu, tpu, tpu.scaling_stiffness(), (True, True)),
        (tpo, tpo, tpo.scaling_stiffness(), (True, True)),
    ]:
        for l in range(1, ml + 1):
            Delta_l_in = basis_in.scaling_indices_on_level(l)
            Delta_l_out = basis_out.scaling_indices_on_level(l)
            eye = np.eye(len(Delta_l_in))
            for i, labda in enumerate(Delta_l_in.asarray()):
                phi_supp = basis_in.scaling_support(labda)
                unit_vec = IndexedVector(Delta_l_in, eye[i, :])
                out = operator.matvec(Delta_l_in, Delta_l_out, unit_vec)
                for mu in Delta_l_out.asarray():
                    supp = phi_supp.intersection(basis_out.scaling_support(mu))
                    true_val = 0.0
                    if supp:
                        true_val = quad(
                            lambda x: basis_in.eval_scaling(
                                labda, x, deriv=deriv[0]) * basis_out.
                            eval_scaling(mu, x, deriv=deriv[1]), supp.a,
                            supp.b)[0]
                    try:
                        assert np.isclose(out[mu], true_val)
                    except AssertionError:
                        print(basis_in.__class__.__name__,
                              basis_out.__class__.__name__, operator,
                              Delta_l_in, Delta_l_out, labda,
                              operator.row(labda), mu, true_val, out[mu])
                        raise


def print_3point_functions():
    x = np.linspace(0, 1, 1025)
    for basis in [
            HaarBasis.uniform_basis(max_level=4),
            HaarBasis.origin_refined_basis(max_level=4),
            OrthonormalDiscontinuousLinearBasis.uniform_basis(max_level=4),
            OrthonormalDiscontinuousLinearBasis.origin_refined_basis(
                max_level=4),
            ThreePointBasis.uniform_basis(max_level=4),
            ThreePointBasis.origin_refined_basis(max_level=4)
    ]:
        for l in range(basis.indices.maximum_level + 1):
            for labda in basis.scaling_indices_on_level(l):
                plt.plot(x, basis.eval_scaling(labda, x), label=labda)
            plt.title("Scaling functions on level {} for {}".format(l, basis.__class__.__name__))
            plt.legend()
            plt.show()

        for l in range(basis.indices.maximum_level + 1):
            for labda in basis.indices.on_level(l):
                plt.plot(x, basis.eval_wavelet(labda, x), label=labda)
            plt.title("Wavelet functions on level {} for {}".format(l, basis.__class__.__name__))
            plt.legend()
            plt.show()


if __name__ == "__main__":
    print_3point_functions()
