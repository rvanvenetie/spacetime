import numpy as np
import scipy.sparse as sp

from .linop import KroneckerLinearOperator


def test_kronecker_linear_operator():
    """ Test that for A and B matrices, (A kron B)x = KronLinOp(A,B)(x). """
    A = sp.random(10, 5, density=0.1)
    B = sp.random(6, 8, density=0.1)
    AkronBmat = sp.kron(A, B)
    AkronBlinop = KroneckerLinearOperator(A, B)
    for _ in range(100):
        v = np.random.rand(5 * 8)
        assert np.allclose(AkronBmat @ v, AkronBlinop @ v)
