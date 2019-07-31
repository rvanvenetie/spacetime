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
                out[labda] = vec.dot(indices_in.indices, self.col(labda))
        else:
            return IndexedVector({
                labda: vec.dot(indices_in.indices, self.col(labda))
                for labda in indices_out
            })
