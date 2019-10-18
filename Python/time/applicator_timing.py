import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from . import applicator, operators
from .sparse_vector import SparseVector
from .three_point_basis import ThreePointBasis


def plot_results(results):
    for basis_name in results:
        plt.plot(range(len(results[basis_name]['N'])), [
            t / n for n, t in zip(results[basis_name]['N'], results[basis_name]
                                  ['apply_time'])
        ],
                 label=basis_name)
    plt.title("Complexity of applying mass matrices")
    plt.xlabel("Maximum level")
    plt.ylabel("Seconds per dof to apply 1 mass matrix")
    plt.grid(which='both')
    plt.legend()
    plt.show()
    return results


@pytest.mark.skip("timing test!")
def test_comparison_implementations():
    results = {}
    try:
        for bla in ['ThreePoint_tree_inplace', 'ThreePoint_tree']:
            results[bla] = {'N': [], 'apply_time': []}
        for level in range(1, 25):
            basis_obj_tree, Lambda_tree = ThreePointBasis.uniform_basis(
                max_level=level)
            applicator_tree = applicator.Applicator(
                basis_obj_tree, operators.mass(basis_obj_tree), Lambda_tree)
            N = len(Lambda_tree)
            vec = SparseVector(Lambda_tree, np.random.rand(N))
            start = time.time()
            applicator_tree.apply(vec)
            apply_time = time.time() - start
            results['ThreePoint_tree']['N'].append(N)
            results['ThreePoint_tree']['apply_time'].append(apply_time)
            print(results)
        return results
    except KeyboardInterrupt:
        return results


@pytest.mark.skip("timing test!")
def test_linear_complexity():
    results = {}
    try:
        for level in range(1, 18):
            basis_obj, Lambda = ThreePointBasis.uniform_basis(max_level=level)
            if not basis_obj.__class__.__name__ in results:
                results[basis_obj.__class__.__name__] = {
                    'N': [],
                    'apply_time': []
                }

            applicator_obj = applicator.Applicator(basis_obj,
                                                   operators.mass(basis_obj),
                                                   Lambda)
            N = len(Lambda)
            vec = SparseVector(Lambda, np.random.rand(N))
            start = time.time()
            applicator_obj.apply(vec)
            apply_time = time.time() - start

            results[basis_obj.__class__.__name__]['N'].append(N)
            results[basis_obj.__class__.__name__]['apply_time'].append(
                apply_time)

            print(results)
        return results
    except KeyboardInterrupt:
        return results


if __name__ == "__main__":
    results = test_comparison_implementations()
    plot_results(results)
    results = test_linear_complexity()
    plot_results(results)
