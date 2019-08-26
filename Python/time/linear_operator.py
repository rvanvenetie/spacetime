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
            for labda in indices_out:
                result[labda] = vec.dot(row_op(labda), indices_in)
        return IndexedVector(result)

    @staticmethod
    def _matvec_inplace(row_op, col_op, indices_in, indices_out, read_from, write_to):
        """ Simple wrapper for matvec and rmatvec. """
        if indices_out is None:
            assert indices_in is not None

            for labda_in in indices_in:
                coeff_in = labda_in.coeff[read_from]
                for labda_out, coeff_out in col_op(labda_in):
                    labda_out.coeff[write_to] += coeff_out * coeff_in

        else:
            for labda_out in indices_out:
                for labda_in, coeff_in in row_op(labda_out):
                    labda_out.coeff[write_to] += labda_in.coeff[read_from] * coeff_in

    def matvec(self, vec, indices_in=None, indices_out=None):
        """ Computes matrix-vector product z = A x.

        By default, calculates z = Ax by taking linear combination of
        columns. If indices_out is specified, it will calculate the result
        using rowwise inner products instead.

        Arguments
            vec: IndexedVector -- input vector
            indices_in: IndexSet -- optional; rows of `vec` to treat as nonzero.
            indices_out: IndexSet -- optional; rows of `out` to set.
        """
        return self._matvec(self.row, self.col, vec, indices_in, indices_out)

    def matvec_inplace(self, indices_in, indices_out, read_from, write_to):
        return self._matvec_inplace(self.row, self.col, indices_in, indices_out, read_from, write_to)

    def rmatvec(self, vec, indices_in=None, indices_out=None):
        """ Computes z = A^T x. """
        return self._matvec(self.col, self.row, vec, indices_in, indices_out)

    def rmatvec_inplace(self, indices_in, indices_out, read_from, write_to):
        return self._matvec_inplace(self.col, self.row, indices_in, indices_out, read_from, write_to)

    def range(self, indices_in):
        """ Returns the range (indices_out) of the operator. """
        return {labda_out for labda_in in indices_in for labda_out, _ in self.col(labda_in)}

    def domain(self, indices_out):
        """ Returns the domain (indices_in) of the oeprator. """
        return {labda_in for labda_out in indices_out for labda_in, _ in self.row(labda_out)}
