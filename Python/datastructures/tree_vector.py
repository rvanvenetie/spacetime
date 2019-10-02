import numpy as np

from .tree import NodeInterface
from .tree_view import MetaRootView, NodeView


class NodeVector(NodeView):
    """ This is a vector on a subtree of an existing underlying tree. """
    def __init__(self, node, value=0.0, parents=None, children=None):
        assert isinstance(node, NodeInterface)
        super().__init__(node, parents=parents, children=children)
        self.value = value

    def copy_data_from(self, other):
        super().copy_data_from(other)
        self.value = other.value

    def __iadd__(self, other):
        """ Shallow `add` operator. """
        self.value += other.value
        return self

    def __imul__(self, x):
        """ Shallow `mul` operator. """
        assert isinstance(x, (int, float, complex)) and not isinstance(x, bool)
        self.value *= x
        return self

    def __repr__(self):
        return 'NV_{}: {}'.format(self.node, self.value)


class MetaRootVector(MetaRootView):
    def __init__(self, metaroot, node_view_cls=NodeVector):
        super().__init__(metaroot=metaroot, node_view_cls=node_view_cls)

    def to_array(self):
        """ Flattens the tree vector into a simple numpy vector. """
        nodes = self.bfs()
        result = np.empty(len(nodes), dtype=float)
        for idx, node in enumerate(nodes):
            result[idx] = node.value
        return result

    def from_array(self, array):
        """ Loads the values from the array in BFS-order into the treevector. """
        nodes = self.bfs()
        assert len(nodes) == len(array)
        for idx, node in enumerate(nodes):
            node.value = array[idx]
        return self
