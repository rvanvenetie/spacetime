from .double_tree import DoubleNode, FrozenDoubleNode


class DoubleNodeVector(DoubleNode):
    """ Extends the double node to incorporate a value. """
    __slots__ = ['value']

    def __init__(self, nodes, value=0, parents=None, children=None):
        super().__init__(nodes=nodes, parents=parents, children=children)
        self.value = value

    def __repr__(self):
        return '({} x {}): {}'.format(self.nodes[0], self.nodes[1], self.value)

    def __iadd__(self, other):
        assert isinstance(other, DoubleNodeVector)
        self.value += other.value


class FrozenDoubleNodeVector(FrozenDoubleNode):
    @property
    def value(self):
        return self.dbl_node.value

    @value.setter
    def value(self, other):
        self.dbl_node.value = other

    def items(self):
        # TODO: This should be removed. Right now, just for compatibility!
        return [(f_node.node, f_node.value) for f_node in self.bfs()]
