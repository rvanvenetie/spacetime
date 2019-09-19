class FunctionNode(NodeAbstract):
    """ This represents a (multilevel) function. """
    def __init__(self, labda, support, parents=None, children=None):
        """  Function for labda = (l, x) living on level l. """
        super().__init__(parents, children)
        self.labda = labda
        self.support = support

    @property
    def level(self):
        return self.labda[0]
