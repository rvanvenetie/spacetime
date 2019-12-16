import time
from abc import ABC, abstractmethod

import numpy as np
import scipy
import scipy.sparse.linalg

from ..datastructures.multi_tree_vector import BlockTreeVector


class ApplicatorInterface(ABC):
    def __init__(self, Lambda_in=None, Lambda_out=None):
        """ Initializes this applicator.

        Args:
          Lambda_*: If set, these must be the indices of the in/output.
        """
        self.Lambda_in = Lambda_in
        self.Lambda_out = Lambda_out

    def shape(self):
        result = [-1, -1]
        if self.Lambda_out:
            result[0] = len(self.Lambda_out.bfs())

        if self.Lambda_in:
            result[1] = len(self.Lambda_in.bfs())

        return tuple(result)

    @abstractmethod
    def apply(self, vec_in, vec_out=None):
        """ Applies this bilinear form to the given input/output.

        Arguments:
            vec_in: a (multi)tree vector holding the linear combination
              of input functions.
            vec_out: Optional. If supported, and set, it should
              hold an *empty* (mult)tree vector for the output functions.

        Returns:
            self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
    @abstractmethod
    def transpose(self):
        """ Returns the transpose of this bilinear form. """
    def __neg__(self):
        """ Returns a negated operator. """
        return ScalarApplicator(self, scalar=-1)

    def __add__(self, other):
        return SumApplicator(self, other)


class SumApplicator(ApplicatorInterface):
    """ Simple wrapper that sums the output of two applicators. """
    def __init__(self, applicator_a, applicator_b):
        assert isinstance(applicator_a, ApplicatorInterface)
        assert isinstance(applicator_b, ApplicatorInterface)
        assert applicator_a.Lambda_in == applicator_b.Lambda_in
        assert applicator_a.Lambda_out == applicator_b.Lambda_out

        super().__init__(Lambda_in=applicator_a.Lambda_in,
                         Lambda_out=applicator_a.Lambda_out)
        self.applicator_a = applicator_a
        self.applicator_b = applicator_b

    def apply(self, *args):
        result = self.applicator_a.apply(*args)
        result += self.applicator_b.apply(*args)
        return result

    def transpose(self):
        return SumApplicator(self.applicator_a.transpose(),
                             self.applicator_b.transpose())


class ScalarApplicator(ApplicatorInterface):
    """ Wrapper that multiples the output of an applicator with a scalar. """
    def __init__(self, applicator, scalar):
        """ Initialize with the applicator whose output is to be negated. """
        assert isinstance(applicator, ApplicatorInterface)
        assert not isinstance(applicator, ScalarApplicator)
        super().__init__(Lambda_in=applicator.Lambda_in,
                         Lambda_out=applicator.Lambda_out)
        self.applicator = applicator
        self.scalar = scalar

    def apply(self, *args):
        result = self.applicator.apply(*args)
        result *= self.scalar
        return result

    def transpose(self):
        return ScalarApplicator(self.applicator.transpose(), self.scalar)


class CompositeApplicator(ApplicatorInterface):
    """ Composes multiple applicators. """
    def __init__(self, applicators):
        """ Initializes this applicator composed of the given applicators.

        First applicators[0] is applied, then applicators[1], etc.. """
        assert isinstance(applicators, (tuple, list))
        for i in range(len(applicators) - 1):
            assert applicators[i].Lambda_out is applicators[i + 1].Lambda_in
        assert all(app.Lambda_out is not None for app in applicators)
        super().__init__(Lambda_in=applicators[0].Lambda_in,
                         Lambda_out=applicators[-1].Lambda_out)
        self.applicators = applicators

    def apply(self, vec_in, **kwargs):
        assert 'vec_out' not in kwargs

        prev_vec = vec_in
        for i, applicator in enumerate(self.applicators):
            prev_vec = applicator.apply(vec_in=prev_vec, **kwargs)
        return prev_vec

    def transpose(self):
        return CompositeApplicator(
            [app.transpose() for app in reversed(self.applicators)])


class BlockApplicator(ApplicatorInterface):
    """ Constructs an applicator for the product space Z_0 x Z_1. """
    def __init__(self, applicators):
        """ Initializes the block-bilinear applicator of the form.

        Args:
           applicators: A 2x2 list of applicators, representing the
             block-linear bilinear form.
        """
        assert 2 == len(applicators) == len(applicators[0]) == len(
            applicators[1])
        # Check that all inputs/outputs are correct
        assert len({applicators[i][0].Lambda_in for i in range(2)}) == 1
        assert len({applicators[i][1].Lambda_in for i in range(2)}) == 1
        assert len({applicators[0][i].Lambda_out for i in range(2)}) == 1
        assert len({applicators[1][i].Lambda_out for i in range(2)}) == 1

        super().__init__(Lambda_in=(applicators[0][0].Lambda_in,
                                    applicators[0][1].Lambda_in),
                         Lambda_out=(applicators[0][0].Lambda_out,
                                     applicators[1][0].Lambda_out))
        self.applicators = applicators

    def shape(self):
        result = [-1, -1]
        if self.Lambda_out[0] and self.Lambda_out[1]:
            result[0] = sum(len(l.bfs()) for l in self.Lambda_out)

        if self.Lambda_in[0] and self.Lambda_in[1]:
            result[1] = sum(len(l.bfs()) for l in self.Lambda_in)

        return tuple(result)

    def apply(self, vec_in):
        """ Applies this block-bilinear form the given input vectors.

        Arguments:
            vec_in: (vec_0, vec_1) a block-vector on Z_0 x Z_1.
        """
        assert isinstance(vec_in, BlockTreeVector)
        out_0 = self.applicators[0][0].apply(vec_in[0])
        out_0 += self.applicators[0][1].apply(vec_in[1])

        out_1 = self.applicators[1][0].apply(vec_in[0])
        out_1 += self.applicators[1][1].apply(vec_in[1])

        return BlockTreeVector([out_0, out_1])

    def transpose(self):
        return BlockApplicator([
            [
                self.applicators[0][0].transpose(),
                self.applicators[1][0].transpose()
            ],
            [
                self.applicators[0][1].transpose(),
                self.applicators[1][1].transpose()
            ],
        ])


class LinearOperatorApplicator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, applicator, input_vec):
        """ Creates a linear operator, given an applicator.

        Args:
          applicator: the applicator this linear operator represents.
          input_vec: an suitable datastructure for holding an input vector.
        """
        super().__init__(float, applicator.shape())
        self.applicator = applicator
        self.input_vec = input_vec
        self.total_time = 0
        self.total_applies = 0

    def _matvec(self, x):
        time_begin = time.process_time()

        self.input_vec.from_array(x)
        result = self.applicator.apply(self.input_vec).to_array()

        self.total_time += time.process_time() - time_begin
        self.total_applies += 1
        return result

    def to_matrix(self):
        return self.matmat(np.eye(self.shape[1]))

    def time_per_dof(self):
        """ Returns an estimated time per dof. """
        return self.total_time / (self.total_applies * self.shape[1])
