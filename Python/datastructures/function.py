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

    @property
    @abstractmethod
    def labda(self):
        pass

    @property
    def level(self):
        return self.labda[0]

    def __repr__(self):
        return "({}, {})".format(*self.labda)


class FunctionNode(FunctionInterface, NodeAbstract):
    """ This represents a (multilevel) function. """
    __slots__ = ['labda']

    def __init__(self, labda, parents=None, children=None):
        """  Function for labda = (l, x) living on level l. """
        super().__init__(parents, children)
        self.labda = labda
