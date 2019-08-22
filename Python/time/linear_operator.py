from indexed_vector import IndexedVector


class LinearOperator(object):
    """ Linear operator. """

    def __init__(self, row, col=None):
        """ Initialize the LinearOperator.

        Arguments:
            row: a function taking an index and returning an IndexedVector,
                being the nonzero entries of this operator at this row. Needed
                for matrix-vector products.
            col (optional): a similar function as `row` but for the column of
                the operator. Needed for z = A^T x.

        """
        self.row = row
        self.col = col

    def matvec(self, indices_in, indices_out, vec, out=None):
        """ Computes matrix-vector product z = A x.

        Arguments
            indices_in: IndexSet -- rows of `vec` to treat as nonzero
            indices_out: IndexSet -- rows of `out` to set
            vec: IndexedVector -- input vector
            out: IndexedVector -- optional; if absent, return out.
        """
        if out:
            for labda in indices_out:
                out[labda] = vec.dot(indices_in, self.row(labda))
        else:
            return IndexedVector({
                labda: vec.dot(indices_in, self.row(labda))
                for labda in indices_out
            })

    def rmatvec(self, indices_in, indices_out, vec, out=None):
        """ Computes z = A^T x. """
        assert self.col
        if out:
            for labda in indices_out:
                out[labda] = vec.dot(indices_in, self.col(labda))
        else:
            return IndexedVector({
                labda: vec.dot(indices_in, self.col(labda))
                for labda in indices_out
            })

    def range(self, indices_in):
        """ Returns the range (indices_out) of the operator. """
        return {labda_out for labda_in in indices_in for labda_out in self.col(labda_in)}

    def domain(self, indices_out):
        """ Returns the domain (indices_in) of the oeprator. """
        return {labda_in for labda_out in indices_out for labda_in in self.row(labda_out)}
