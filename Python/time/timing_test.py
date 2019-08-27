import cProfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

import applicator
import applicator_inplace
import basis
import operators
from old.applicator import Applicator
from old.haar_basis import HaarBasis
from old.orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from old.three_point_basis import ThreePointBasis
from sparse_vector import SparseVector


def plot_results(results):
    for basis_name in results:
        #plt.loglog(results[basis_name]['N'],
        #           results[basis_name]['apply_time'],
        #           basex=2,
        #           basey=2,
        #           label=basis_name)
        plt.plot(
            range(len(results[basis_name]['N'])), [
                t / n for n, t in zip(results[basis_name]['N'],
                                      results[basis_name]['apply_time'])
            ],
            label=basis_name)
    #plt.loglog(results['ThreePointBasis']['N'], [
    #    results['ThreePointBasis']['apply_time'][-1] /
    #    results['ThreePointBasis']['N'][-1] * N
    #    for N in results['ThreePointBasis']['N']
    #],
    #           'k--',
    #           label='linear')
    #plt.plot(results['ThreePointBasis']['N'], [
    #    results['ThreePointBasis']['apply_time'][-1] /
    #    results['ThreePointBasis']['N'][-1]**2 * N**2
    #    for N in results['ThreePointBasis']['N']
    #],
    #           'k-.',
    #           label='quadratic')
    plt.title("Complexity of applying mass matrices")
    plt.xlabel("Maximum level")
    #plt.ylabel("Seconds to apply 1 mass matrix")
    plt.ylabel("Seconds per dof to apply 1 mass matrix")
    plt.grid(which='both')
    plt.legend()
    plt.show()
    return results


@pytest.mark.skip("timing test!")
def test_comparison_implementations():
    results = {}
    try:
        for bla in [
                'ThreePoint_tree_inplace', 'ThreePoint_tree', 'ThreePoint'
        ]:
            results[bla] = {'N': [], 'apply_time': []}
        for level in range(1, 25):
            basis_obj_tree, Lambda_tree = basis.ThreePointBasis.uniform_basis(
                max_level=level)
            applicator_tree = applicator.Applicator(
                basis_obj_tree, operators.mass(basis_obj_tree), Lambda_tree)
            N = len(Lambda_tree)
            vec = SparseVector(Lambda_tree, np.random.rand(N))
            start = time.time()
            res = applicator_tree.apply(vec)
            apply_time = time.time() - start
            results['ThreePoint_tree']['N'].append(N)
            results['ThreePoint_tree']['apply_time'].append(apply_time)

            basis_obj_tree_inplace, Lambda_tree_inplace = basis.ThreePointBasis.uniform_basis(
                max_level=level)
            applicator_tree_inplace = applicator_inplace.Applicator(
                basis_obj_tree_inplace, operators.mass(basis_obj_tree),
                Lambda_tree_inplace)
            N = len(Lambda_tree_inplace)
            vec = SparseVector(Lambda_tree_inplace, np.random.rand(N))
            start = time.time()
            res = applicator_tree_inplace.apply(vec)
            apply_time = time.time() - start
            results['ThreePoint_tree_inplace']['N'].append(N)
            results['ThreePoint_tree_inplace']['apply_time'].append(apply_time)

            basis_old = ThreePointBasis.uniform_basis(max_level=level)
            applicator_old = Applicator(basis_old, basis_old.scaling_mass(),
                                        basis_old.indices)
            N = len(basis_old.indices)
            vec = SparseVector(basis_old.indices, np.random.rand(N))
            start = time.time()
            res = applicator_old.apply(vec)
            apply_time = time.time() - start

            results['ThreePoint']['N'].append(N)
            results['ThreePoint']['apply_time'].append(apply_time)
            print(results)
        return results
    except KeyboardInterrupt:
        return results


@pytest.mark.skip("timing test!")
def test_linear_complexity_tree():
    results = {}
    try:
        for level in range(17, 20):
            basis_obj, Lambda = basis.ThreePointBasis.uniform_basis(
                max_level=level)
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
            res = applicator_obj.apply(vec)
            apply_time = time.time() - start

            results[basis_obj.__class__.__name__]['N'].append(N)
            results[basis_obj.__class__.__name__]['apply_time'].append(
                apply_time)

            print(results)
        return results
    except KeyboardInterrupt:
        return results


@pytest.mark.skip("timing test!")
def test_linear_complexity():
    results = {}
    try:
        for level in range(1, 100):
            for basis in [
                    #HaarBasis.uniform_basis(max_level=level),
                    #OrthonormalDiscontinuousLinearBasis.uniform_basis(
                    #    max_level=level - 1),
                    ThreePointBasis.uniform_basis(max_level=level)
            ]:
                if not basis.__class__.__name__ in results:
                    results[basis.__class__.__name__] = {
                        'N': [],
                        'apply_time': []
                    }

                applicator = Applicator(basis, basis.scaling_mass(),
                                        basis.indices)
                N = len(basis.indices)
                vec = SparseVector(basis.indices, np.random.rand(N))
                start = time.time()
                res = applicator.apply(vec)
                apply_time = time.time() - start

                results[basis.__class__.__name__]['N'].append(N)
                results[basis.__class__.__name__]['apply_time'].append(
                    apply_time)

                print(results)

    except KeyboardInterrupt:
        return results
    finally:
        return results


if __name__ == "__main__":
    #results = test_comparison_implementations()
    results = test_linear_complexity_tree()
    #results = test_linear_complexity()
    #cProfile.run('results = test_linear_complexity()', sort='tottime')
    plot_results(results)
