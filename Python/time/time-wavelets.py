import numpy as np

from index import IndexSet, IndexedVector
from basis import HaarBasis

if __name__ == "__main__":
    basis = HaarBasis()
    for Lambda in [
            basis.uniform_wavelet_indices(max_level=3),
            basis.origin_refined_wavelet_indices(max_level=3)
    ]:
        c = IndexedVector(Lambda, np.ones(Lambda.len()))
        operator = basis.singlescale_mass
        print(basis.apply_operator(operator, Lambda, c).vector)
