from abc import abstractmethod

from .tree import NodeAbstract


class FunctionNode(NodeAbstract):
    """ This represents a (multilevel) function. """
    def __init__(self, labda, parents=None, children=None):
        """  Function for labda = (l, x) living on level l. """
        super().__init__(parents, children)
        self.labda = labda

    @property
    @abstractmethod
    def support(self):
        pass

    @property
    def level(self):
        return self.labda[0]

    def __repr__(self):
        return "({}, {})".format(*self.labda)
