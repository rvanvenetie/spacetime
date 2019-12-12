from scipy.sparse.linalg import LinearOperator
import numpy as np


def KroneckerLinearOperator(R1, R2):
    N, K = R1.shape
    M, L = R2.shape

    def matvec(x):
        X = x.reshape(K, L)
        return R2.dot(R1.dot(X).T).T.reshape(-1)

    return LinearOperator(matvec=matvec, shape=(N * M, K * L))
