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

    def inner_quad(self, g, deriv=False, order=4):
        """ Computes <g, self> or <g, grad self> by quadrature.
        
        Arguments:
            g: the function to take inner products with. Takes numpy array x.
            deriv: whether to evaluate the derivative (gradient) of `self`.
            order: the polynomial order to be reproduced by the quadrature rule.
        """
        raise NotImplemented(
            'The inner_quad function is not (yet) implemented')
