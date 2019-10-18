from abc import ABC, abstractmethod


class FunctionInterface(ABC):
    """ This represents a (multilevel) function. """
    __slots__ = []
    order = None

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
        raise NotImplementedError

    def inner_quad(self, g, g_order=4, deriv=False):
        """ Computes <g, self> or <g, grad self> by quadrature.
        
        Arguments:
            g: the function to take inner products with. Takes numpy array x.
            g_order: the polynomial order of g.
            deriv: whether to evaluate the derivative (gradient) of `self`.
        """
        raise NotImplementedError
