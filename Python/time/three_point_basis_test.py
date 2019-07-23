from three_point_basis import ThreePointBasis
from index_set import IndexSet
import matplotlib.pyplot as plt
import numpy as np


def print_functions():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    x = np.linspace(0, 1, 1025)
    for labda in basis.ss_indices[3]:
        plt.plot(x, basis.eval_scaling(labda, x), label=labda)
    plt.legend()
    plt.show()

    for labda in multiscale_indices:
        plt.plot(x, basis.eval_wavelet(labda, x), label=labda)
    plt.legend()
    plt.show()


def test_singlescale_indices():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    assert basis.ss_indices[1].indices == {(1, 0), (1, 1), (1, 2)}
    assert basis.ss_indices[2].indices == {(2, 0), (2, 1), (2, 2), (2, 3),
                                           (2, 4)}
    assert basis.ss_indices[3].indices == {(3, 0), (3, 1), (3, 2), (3, 4),
                                           (3, 6), (3, 7), (3, 8)}


def test_siblings_etc():
    multiscale_indices = IndexSet({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
                                   (3, 0), (3, 3)})
    basis = ThreePointBasis(multiscale_indices)
    for index in multiscale_indices:
        for ss_index in basis.wavelet_siblings(index):
            assert basis.wavelet_support(index).intersects(
                basis.scaling_support(ss_index))


if __name__ == "__main__":
    print_functions()
