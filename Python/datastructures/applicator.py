from abc import ABC, abstractmethod


class ApplicatorInterface(ABC):
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
        pass

    @abstractmethod
    def transpose(self):
        """ Returns the transpose of this bilinear form. """
        pass

    def __neg__(self):
        """ Returns a negated operator. """
        return ScalarApplicator(self, -1)

    def __add__(self, other):
        return SumApplicator(self, other)


class SumApplicator(ApplicatorInterface):
    """ Simple wrapper that sums the output of two applicators. """
    def __init__(self, applicator_a, applicator_b):
        assert isinstance(applicator_a, ApplicatorInterface)
        assert isinstance(applicator_b, ApplicatorInterface)
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
    """ Simple wrapper that multiples the output of an applicator with a scalar. """
    def __init__(self, applicator, scalar):
        """ Initialize with the applicator whose output is to be negated. """
        assert isinstance(applicator, ApplicatorInterface)
        assert not isinstance(applicator, ScalarApplicator)
        self.applicator = applicator
        self.scalar = scalar

    def apply(self, *args):
        result = self.applicator.apply(*args)
        result *= self.scalar
        return result

    def transpose(self):
        return ScalarApplicator(self.applicator.transpose(), self.scalar)


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
        self.applicators = applicators

    def apply(self, vec):
        """ Applies this block-bilinear form the given input vectors.

        Arguments:
            vec: (vec_0, vec_1) a block-vector on Z_0 x Z_1.
        """
        out_0 = self.applicators[0][0].apply(vec[0])
        out_0 += self.applicators[0][1].apply(vec[1])

        out_1 = self.applicators[1][0].apply(vec[0])
        out_1 += self.applicators[1][1].apply(vec[1])

        return (out_0, out_1)

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
