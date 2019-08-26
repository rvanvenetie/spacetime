import cProfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

import applicator_tree
import applicator_tree_inplace
import basis_tree
from applicator import Applicator
from haar_basis import HaarBasis
from indexed_vector import IndexedVector
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis


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


def test_mother_fucker():
    results = {}
    try:
        for bla in [
                'ThreePoint_tree_inplace', 'ThreePoint_tree', 'ThreePoint'
        ]:
            results[bla] = {'N': [], 'apply_time': []}
        for level in range(1, 25):
            basis_obj_tree, Lambda_tree = basis_tree.ThreePointBasis.uniform_basis(
                max_level=level)
            applicator = applicator_tree.Applicator(
                basis_obj_tree, basis_obj_tree.scaling_mass(), Lambda_tree)
            N = len(Lambda_tree)
            vec = IndexedVector(Lambda_tree, np.random.rand(N))
            start = time.time()
            res = applicator.apply(vec)
            apply_time = time.time() - start
            results['ThreePoint_tree']['N'].append(N)
            results['ThreePoint_tree']['apply_time'].append(apply_time)

            basis_obj_tree_inplace, Lambda_tree_inplace = basis_tree.ThreePointBasis.uniform_basis(
                max_level=level)
            applicator = applicator_tree_inplace.Applicator(
                basis_obj_tree_inplace, basis_obj_tree_inplace.scaling_mass(),
                Lambda_tree_inplace)
            N = len(Lambda_tree_inplace)
            vec = IndexedVector(Lambda_tree_inplace, np.random.rand(N))
            start = time.time()
            res = applicator.apply(vec)
            apply_time = time.time() - start
            results['ThreePoint_tree_inplace']['N'].append(N)
            results['ThreePoint_tree_inplace']['apply_time'].append(apply_time)

            basis = ThreePointBasis.uniform_basis(max_level=level)
            applicator = Applicator(basis, basis.scaling_mass(), basis.indices)
            N = len(basis.indices)
            vec = IndexedVector(basis.indices, np.random.rand(N))
            start = time.time()
            res = applicator.apply(vec)
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
            for basis, Lambda in [
                    basis_tree.ThreePointBasis.uniform_basis(max_level=level)
            ]:
                if not basis.__class__.__name__ in results:
                    results[basis.__class__.__name__] = {
                        'N': [],
                        'apply_time': []
                    }

                applicator = applicator_tree_inplace.Applicator(
                    basis, basis.scaling_mass(), Lambda)
                N = len(Lambda)
                vec = IndexedVector(Lambda, np.random.rand(N))
                start = time.time()
                res = applicator.apply(vec)
                apply_time = time.time() - start

                results[basis.__class__.__name__]['N'].append(N)
                results[basis.__class__.__name__]['apply_time'].append(
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
                vec = IndexedVector(basis.indices, np.random.rand(N))
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
    results = test_mother_fucker()
    #results = test_linear_complexity_tree()
    #results = test_linear_complexity()
    #cProfile.run('results = test_linear_complexity()', sort='tottime')
    plot_results(results)
