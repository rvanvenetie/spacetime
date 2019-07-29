from indexed_vector import IndexedVector


class LinearOperator(object):
    """ Linear operator. """

    def __init__(self, row_fn, col_fn=None):
        """ Initialize. """
        self.row = row_fn
        self.col = col_fn

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
