from three_point_basis import ThreePointBasis, ms2ss, ss2ms
from index_set import IndexSet
from indexed_vector import IndexedVector

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def print_functions():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    x = np.linspace(0, 1, 1025)
    for labda in basis.scaling_indices_on_level(3):
        plt.plot(x, basis.eval_scaling(labda, x), label=labda)
    plt.legend()
    plt.show()

    for labda in multiscale_indices:
        plt.plot(x, basis.eval_wavelet(labda, x), label=labda)
    plt.legend()
    plt.show()


def test_ss2ms():
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


def test_singlescale_indices():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    assert basis.scaling_indices_on_level(1).indices == {(1, 0), (1, 1),
                                                         (1, 2)}
    assert basis.scaling_indices_on_level(2).indices == {(2, 0), (2, 1),
                                                         (2, 2), (2, 3),
                                                         (2, 4)}
    assert basis.scaling_indices_on_level(3).indices == {
        (3, 0), (3, 1), (3, 2), (3, 4), (3, 6), (3, 7), (3, 8)
    }


def test_singlescale_mass():
    """ Use quadrature to test the singlescale mass matrix. """
    for basis in [
            ThreePointBasis.uniform_basis(max_level=5),
            ThreePointBasis.origin_refined_basis(max_level=5)
    ]:
        for l, Delta_l in enumerate(basis.ss_indices):
            eye = np.eye(len(Delta_l))
            for i, labda in enumerate(sorted(Delta_l)):
                phi_supp = basis.scaling_support(labda)
                unit_vec = IndexedVector(Delta_l, eye[i, :])
                out = basis.singlescale_mass(l=l,
                                             Pi=Delta_l,
                                             Pi_A=Delta_l,
                                             d=unit_vec)
                for mu in sorted(Delta_l):
                    supp = phi_supp.intersection(basis.scaling_support(mu))
                    if supp:
                        assert np.isclose(
                            out[mu],
                            quad(
                                lambda x: basis.eval_scaling(labda, x) * basis.
                                eval_scaling(mu, x), supp.a, supp.b)[0])
                    else:
                        assert np.isclose(out[mu], 0.0)


def test_siblings_etc():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3), (4, 0)})
    basis = ThreePointBasis(multiscale_indices)
    for index in multiscale_indices:
        assert basis.wavelet_siblings(index) == sorted([
            i for i in basis.scaling_indices_on_level(index[0])
            if basis.wavelet_support(index).contains(basis.scaling_support(i))
        ])

    for level, ss_indices_at_level in enumerate(basis.ss_indices):
        if level > 0:
            for index in sorted(ss_indices_at_level):
                assert basis.scaling_parents(index) == sorted([
                    i for i in basis.scaling_indices_on_level(level - 1)
                    if index in basis.scaling_children(i)
                ])
        if level < basis.indices.maximum_level():
            for index in sorted(ss_indices_at_level):
                assert basis.scaling_children(index) == sorted([
                    i for i in basis.scaling_indices_on_level(level + 1)
                    if index in basis.scaling_parents(i)
                ])
        for index in sorted(ss_indices_at_level):
            assert basis.scaling_siblings(index) == sorted([
                i for i in multiscale_indices
                if i[0] == level and index in basis.wavelet_siblings(i)
            ])


def test_basis_PQ():
    x = np.linspace(0, 1, 1025)
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3), (4, 0), (4, 7)})
    basis = ThreePointBasis(multiscale_indices)
    for l in range(1, multiscale_indices.maximum_level()):
        Pi_B = basis.scaling_indices_on_level(l - 1)
        Pi_bar = basis.scaling_indices_on_level(l)
        eye = np.eye(len(Pi_B))
        for i, mu in enumerate(sorted(Pi_B.indices)):
            # Write phi_mu on lv l-1 as combination of scalings on lv l.
            vec = IndexedVector(Pi_B, eye[i, :])
            res = basis.apply_P(Pi_B, Pi_bar, vec)
            inner = np.sum([
                basis.eval_scaling(labda, x) * res[labda]
                for labda in res.keys()
            ],
                           axis=0)
            assert np.allclose(inner, basis.eval_scaling(mu, x))

        Lambda_l = basis.wavelet_indices_on_level(l)
        for i, mu in enumerate(sorted(Lambda_l.indices)):
            # Write psi_mu on lv l as combination of scalings on lv l.
            vec = IndexedVector(Lambda_l, eye[i, :])
            res = basis.apply_Q(Lambda_l, Pi_bar, vec)
            inner = np.sum([
                basis.eval_scaling(labda, x) * res[labda]
                for labda in res.keys()
            ],
                           axis=0)
            try:
                assert np.allclose(inner, basis.eval_wavelet(mu, x))
            except AssertionError:
                plt.plot(x, inner)
                plt.plot(x, basis.eval_wavelet(mu, x))
                plt.show()
                assert False


if __name__ == "__main__":
    print_functions()
