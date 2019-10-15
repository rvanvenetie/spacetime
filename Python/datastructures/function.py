from abc import ABC, abstractmethod

from .tree import NodeAbstract


class FunctionInterface(ABC):
    """ This represents a (multilevel) function. """
    @property
    @abstractmethod
    def support(self):
        pass

    @support.setter
    @abstractmethod
    def support(self, value):
        pass

    def eval(self, x, deriv=False):
        """ Evaluates this function at the given coordinate. """
        raise NotImplemented('The eval function is not (yet) implemented')

    @abstractmethod
    def inner_quad(self, g, order=4):
        """ Computes the inner product with `g` using quadrature. """
        pass
