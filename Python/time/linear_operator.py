from collections import defaultdict

from .sparse_vector import SparseVector


class LinearOperator(object):
    """ Linear operator. """
    def __init__(self, row, col=None):
        """ Initialize the LinearOperator.

        Arguments:
            row: a function taking an index and returning a list of tuples,
                being the nonzero entries of this operator at this row. Needed
                for matrix-vector products.
            col (optional): a similar function as `row` but for the column of
                the operator. Needed for z = A^T x. If col = None, it is assumed
                the linear operator is self-adjoint, i.e. row == col.

        """
        self.row = row
        self.col = col if col else row

    @staticmethod
    def _matvec(row_op, col_op, vec, indices_in, indices_out):
        """ Simple wrapper for matvec and rmatvec. """
        result = defaultdict(float)
        if indices_out is None:
            if indices_in is None: indices_in = vec.keys()

            for labda_in in indices_in:
                coeff_in = vec[labda_in]
                for labda_out, coeff_out in col_op(labda_in):
                    result[labda_out] += coeff_out * coeff_in
        else:
            for labda_out in indices_out:
                for labda_in, coeff_in in row_op(labda_out):
                    result[labda_out] += coeff_in * vec[labda_in]
        return SparseVector(result)

    @staticmethod
    def _matvec_inplace(row_op, col_op, indices_in, indices_out, read, write):
        """ Performs inplace matvec for vectors living on index sets. """
        if indices_out is None:
            assert indices_in is not None

            for labda_in in indices_in:
                coeff_in = labda_in.coeff[read]
                for labda_out, coeff_out in col_op(labda_in):
                    labda_out.coeff[write] += coeff_out * coeff_in
        else:
            assert indices_in is None
            for labda_out in indices_out:
                for labda_in, coeff_in in row_op(labda_out):
                    labda_out.coeff[write] += labda_in.coeff[read] * coeff_in

    def matvec(self, vec, indices_in=None, indices_out=None):
        """ Computes matrix-vector product z = A x.

        By default, calculates z = Ax by taking linear combination of
        columns. If indices_out is specified, it will calculate the result
        using rowwise inner products instead.

        Arguments
            vec: dict -- input vector
            indices_in: IndexSet -- optional; rows of `vec` to treat as nonzero.
            indices_out: IndexSet -- optional; rows of `out` to set.
        """
        return self._matvec(self.row, self.col, vec, indices_in, indices_out)

    def matvec_inplace(self, indices_in, indices_out, read, write):
        """ Performs an in-place matvec. """
        return self._matvec_inplace(self.row, self.col, indices_in,
                                    indices_out, read, write)

    def rmatvec(self, vec, indices_in=None, indices_out=None):
        """ Computes z = A^T x. """
        return self._matvec(self.col, self.row, vec, indices_in, indices_out)

    def rmatvec_inplace(self, indices_in, indices_out, read, write):
        return self._matvec_inplace(self.col, self.row, indices_in,
                                    indices_out, read, write)

    def range(self, indices_in):
        """ Returns the range (indices_out) of the operator. """
        return {
            labda_out
            for labda_in in indices_in for labda_out, _ in self.col(labda_in)
        }

    def domain(self, indices_out):
        """ Returns the domain (indices_in) of the oeprator. """
        return {
            labda_in
            for labda_out in indices_out for labda_in, _ in self.row(labda_out)
        }
