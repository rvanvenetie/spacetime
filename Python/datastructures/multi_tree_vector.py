from abc import ABC, abstractmethod

import numpy as np

from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree


class MultiNodeVectorInterface(MultiNodeViewInterface):
    """ Extends the multinode interfac a value. """
    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        pass

    def _deep_sum(self, other):
        assert isinstance(other, MultiNodeVectorInterface)

        def call_sum(my_node, other_node):
            my_node.value += other_node.value

        self._union(other, call_postprocess=call_sum)
        return self

    def __repr__(self):
        return '{}: {}'.format(super().__repr__(), self.value)


class MultiNodeVector(MultiNodeVectorInterface, MultiNodeView):
    __slots__ = ['value']

    def __init__(self, nodes, value=0, parents=None, children=None):
        super().__init__(nodes=nodes, parents=parents, children=children)
        self.value = value


class MultiTreeVector(MultiTree):
    def to_array(self):
        """ Transforms a double tree vector to a numpy vector.  """
        return np.array([node.value for node in self.bfs()], dtype=float)

    def from_array(self, array):
        """ Loads the values from the array in BFS-order into the multi treevector. """
        nodes = self.bfs()
        assert len(nodes) == len(array)
        for idx, node in enumerate(nodes):
            node.value = array[idx]
        return self

    def deep_copy(self,
                  mlt_node_cls=None,
                  mlt_tree_cls=None,
                  call_postprocess=None):
        """ Copies the current multitree. """
        if call_postprocess is None:

            def call_copy(new_node, old_node):
                new_node.value = old_node.value

            call_postprocess = call_copy
        return super().deep_copy(mlt_node_cls, mlt_tree_cls, call_postprocess)

    def __iadd__(self, other):
        """ Add two double trees. """
        assert isinstance(other, MultiTreeVector)
        self.root._deep_sum(other.root)
        return self

    def __imul__(self, x):
        """ Recursive `mul` operator. """
        for node in self.bfs():
            node.value *= x
        return self