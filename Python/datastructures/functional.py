from abc import ABC, abstractmethod


class FunctionalInterface(ABC):
    @abstractmethod
    def eval(self, Lambda_out):
        """ Evaluates this functional on the given index set.  """

    def __neg__(self):
        """ Returns a negated operator. """
        return ScalarFunctional(self, scalar=-1)

    def __add__(self, other):
        return SumFunctional([self, other])


class SumFunctional(FunctionalInterface):
    def __init__(self, functionals):
        assert isinstance(functionals, (tuple, list))
        assert len(functionals)
        assert all(
            isinstance(functional, FunctionalInterface)
            for functional in functionals)

        self.functionals = functionals

    def eval(self, Lambda_out):
        result = self.functionals[0].eval(Lambda_out)
        for functional in self.functionals[1:]:
            result += functional.eval(Lambda_out)
        return result


class ScalarFunctional(FunctionalInterface):
    def __init__(self, functional, scalar):
        self.functional = functional
        self.scalar = scalar

    def eval(self, Lambda_out):
        result = self.functional.eval(Lambda_out)
        result *= self.scalar
        return result
