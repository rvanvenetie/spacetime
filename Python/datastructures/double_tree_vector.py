from .double_tree import DoubleNode, FrozenDoubleNode


class DoubleNodeVector(DoubleNode):
    """ Extends the double node to incorporate a value. """
    __slots__ = ['value']

    def __init__(self, nodes, value=0, parents=None, children=None):
        super().__init__(nodes=nodes, parents=parents, children=children)
        self.value = value

    def __repr__(self):
        return '({} x {}): {}'.format(self.nodes[0], self.nodes[1], self.value)


class FrozenDoubleNodeVector(FrozenDoubleNode):
    @property
    def value(self):
        return self.dbl_node.value
