from applicator import Applicator

from haar_basis import HaarBasis
from orthonormal_basis import OrthonormalDiscontinuousLinearBasis
from three_point_basis import ThreePointBasis
from indexed_vector import IndexedVector

import matplotlib.pyplot as plt
import time
import numpy as np
import cProfile
import pytest


def plot_results(results):
    for basis_name in results:
        plt.plot(results[basis_name]['N'],
                 results[basis_name]['apply_time'],
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
    plt.legend()
    plt.show()
    return results


@pytest.mark.skip("timing test!")
def test_linear_complexity():
    results = {}
    try:
        for level in range(1, 16):
            for basis in [
                    HaarBasis.uniform_basis(max_level=level),
                    OrthonormalDiscontinuousLinearBasis.uniform_basis(
                        max_level=level - 1),
                    ThreePointBasis.uniform_basis(max_level=level)
            ]:
                if not basis.__class__.__name__ in results:
                    results[basis.__class__.__name__] = {
                        'N': [],
                        'apply_time': []
                    }

                applicator = Applicator(basis, basis.singlescale_mass,
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
    cProfile.run('results = test_linear_complexity()', sort='tottime')
    plot_results(results)
