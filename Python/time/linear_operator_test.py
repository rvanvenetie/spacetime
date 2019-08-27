from collections import defaultdict

import numpy as np

from sparse_vector import SparseVector


def check_linop_transpose(linop, indices_in, indices_out):
    A = np.zeros([len(indices_in), len(indices_out)])
    AT = np.zeros([len(indices_out), len(indices_in)])

    for i, mu in enumerate(indices_in):
        vec = SparseVector({mu: 1})
        A[i, :] = linop.matvec(vec, indices_in,
                               indices_out).asarray(indices_out)
    for i, mu in enumerate(indices_out):
        vec = SparseVector({mu: 1})
        AT[i, :] = linop.rmatvec(vec, indices_out,
                                 indices_in).asarray(indices_in)

    assert np.allclose(A.T, AT)
    assert np.allclose(AT @ A, (AT @ A).T)
