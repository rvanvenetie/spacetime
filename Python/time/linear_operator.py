from indexed_vector import IndexedVector
from collections import defaultdict


class LinearOperator(object):
    """ Linear operator. """

    def __init__(self, row, col=None):
        """ Initialize the LinearOperator.

        Arguments:
            row: a function taking an index and returning an IndexedVector,
                being the nonzero entries of this operator at this row. Needed
                for matrix-vector products.
            col (optional): a similar function as `row` but for the column of
                the operator. Needed for z = A^T x. If col = None, it is assumed
                the linear operator is self-adjoint, i.e. row == col.

        """
        self.row = row
        self.col = col if col else row

    def matvec(self, vec, indices_in=None, indices_out=None):
        """ Computes matrix-vector product z = A x.

        Arguments
            vec: IndexedVector -- input vector
            indices_in: IndexSet -- optional; rows of `vec` to treat as nonzero.
            indices_out: IndexSet -- optional; rows of `out` to set.
        """
        result = defaultdict(float)
        if indices_in is None: indices_in = vec.keys()

        if indices_out is None:
            # Simply calculate the Ax for all indices_in.
            for labda_in in indices_in:
                coeff_in = vec[labda_in]
                for labda_out, coeff_out in self.col(labda_in):
                    result[labda_out] += coeff_out * coeff_in
        else:
            for labda in indices_out:
                result[labda] = vec.dot(indices_in, dict(self.row(labda)))

        return IndexedVector(result)

    def rmatvec(self, vec, indices_in=None, indices_out=None):
        """ Computes z = A^T x. """
        assert self.col
        result = {}
        for labda in indices_out:
            result[labda] = vec.dot(indices_in, dict(self.col(labda)))
        return IndexedVector(result)

    def range(self, indices_in):
        """ Returns the range (indices_out) of the operator. """
        return {labda_out for labda_in in indices_in for labda_out, _ in self.col(labda_in)}

    def domain(self, indices_out):
        """ Returns the domain (indices_in) of the oeprator. """
        return {labda_in for labda_out in indices_out for labda_in, _ in self.row(labda_out)}
