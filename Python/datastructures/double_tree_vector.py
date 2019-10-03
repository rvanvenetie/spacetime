import numpy as np

from .double_tree import DoubleNode, DoubleTree, FrozenDoubleNode


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
        return self


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

    def to_array(self):
        """ Flattens the tree vector rooted here into a simple numpy vector. """
        return np.array([node.value for node in self.bfs()])

    def from_array(self, array):
        """ Loads the values from the array in BFS-order into the treevector. """
        nodes = self.bfs()
        assert len(nodes) == len(array)
        for idx, node in enumerate(nodes):
            node.value = array[idx]
        return self


class DoubleTreeVector(DoubleTree):
    def __init__(self, root, frozen_dbl_cls=FrozenDoubleNodeVector):
        if isinstance(root, tuple): root = DoubleNodeVector(root)
        super().__init__(root=root, frozen_dbl_cls=frozen_dbl_cls)

    def to_array(self):
        """ Transforms a double tree vector to a numpy vector.  """
        return np.array([
            psi_1.value for psi_0 in self.project(0).bfs()
            for psi_1 in psi_0.frozen_other_axis().bfs()
        ])

    def from_array(self, array):
        """ Loads the values from an numpy array into this double vector. """
        i = 0
        for psi_0 in self.project(0).bfs():
            for psi_1 in psi_0.frozen_other_axis().bfs():
                psi_1.value = array[i]
                i += 1

    def __iadd__(self, other):
        """ Add two double trees, assuming they have the *same* structure. """
        assert isinstance(other, DoubleTreeVector)
        my_nodes = self.bfs()
        other_nodes = other.bfs()
        assert len(my_nodes) == len(other_nodes)
        for my_node, other_node in zip(my_nodes, other_nodes):
            assert my_node.nodes == other_node.nodes
            my_node += other_node
        return self
