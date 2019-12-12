from scipy.sparse.linalg import LinearOperator


def KroneckerLinearOperator(R1, R2):
    """ Create LinOp that applies kron(A,B)x without explicit construction. """
    N, K = R1.shape
    M, L = R2.shape

    def matvec(x):
        X = x.reshape(K, L)
        return R2.dot(R1.dot(X).T).T.reshape(-1)

    return LinearOperator(matvec=matvec, shape=(N * M, K * L))
