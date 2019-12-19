from abc import abstractmethod

import numpy as np

from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree


class MultiNodeVectorInterface(MultiNodeViewInterface):
    """ Extends the multinode interface with a value. """
    __slots__ = []

    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        pass

    def __repr__(self):
        return '{}: {}'.format(super().__repr__(), self.value)


class MultiNodeVector(MultiNodeVectorInterface, MultiNodeView):
    __slots__ = ['value']

    def __init__(self, nodes, parents=None, children=None):
        super().__init__(nodes=nodes, parents=parents, children=children)
        self.value = 0.0


class MultiTreeVector(MultiTree):
    def to_array(self):
        """ Transforms a multi tree vector to a numpy vector, in kron order. """
        return np.array([node.value for node in self.bfs_kron()], dtype=float)

    def from_array(self, array):
        """ Loads the values from an array in bfs kron order. """
        nodes = self.bfs_kron()
        assert len(nodes) == len(array)
        for idx, node in enumerate(nodes):
            node.value = array[idx]
        return self

    def deep_copy(self,
                  mlt_tree_cls=None,
                  mlt_node_cls=None,
                  call_postprocess=None):
        """ Copies the current multitree. """
        if call_postprocess is None:
            call_postprocess = lambda new, old: setattr(
                new, 'value', old.value)
        return super().deep_copy(mlt_tree_cls=mlt_tree_cls,
                                 mlt_node_cls=mlt_node_cls,
                                 call_postprocess=call_postprocess)

    def axpy(self, other, scalar_mult=1):
        """ Implementation of `self += scalar_mult * other` """
        def call_add(my_node, other_node):
            my_node.value += other_node.value * scalar_mult

        if isinstance(other, MultiTreeVector):
            self.root._union(other.root, call_postprocess=call_add)
        else:
            self.root._union(other, call_postprocess=call_add)

        return self

    def sum(self):
        return sum(nv.value for nv in self.bfs())

    def norm(self):
        """ Returns the l2 norm of this vector. """
        return np.linalg.norm(self.to_array(), 2)

    def reset(self):
        """ Resets all the values in the underlying tree to zero. """
        for node in self.bfs():
            node.value = 0
        return self

    def __iadd__(self, other):
        """ Add two double trees. """
        return self.axpy(other)

    def __isub__(self, other):
        """ Subtract a double tree. """
        return self.axpy(other, -1)

    def __imul__(self, x):
        """ Recursive `mul` operator. """
        if isinstance(x, MultiTreeVector):

            my_nodes = self.bfs()
            x_nodes = x.bfs()
            assert len(my_nodes) == len(x_nodes)
            for my_node, x_node in zip(my_nodes, x_nodes):
                assert my_node.nodes == x_node.nodes
                my_node.value *= x_node.value
        else:
            for node in self.bfs():
                node.value *= x

        return self


class BlockTreeVector:
    def __init__(self, vecs):
        assert isinstance(vecs, (tuple, list))
        assert all(isinstance(vec, MultiTreeVector) for vec in vecs)
        self.vecs = vecs

    def __iter__(self):
        return iter(self.vecs)

    def __getitem__(self, i):
        return self.vecs[i]

    def __isub__(self, other):
        assert isinstance(other, BlockTreeVector) and len(other.vecs) == len(
            self.vecs)
        for i, vec in enumerate(self.vecs):
            vec -= other[i]
        return self

    def __iadd__(self, other):
        assert isinstance(other, BlockTreeVector) and len(other.vecs) == len(
            self.vecs)
        for i, vec in enumerate(self.vecs):
            vec += other[i]
        return self

    def to_array(self):
        return np.concatenate([vec.to_array() for vec in self.vecs])

    def from_array(self, arr):
        lengths = [len(vec.bfs()) for vec in self.vecs]
        assert len(arr) == sum(lengths)
        splitpoints = np.cumsum(lengths)
        arrays = np.split(arr, splitpoints)
        for i, vec in enumerate(self.vecs):
            vec.from_array(arrays[i])

    def bfs(self):
        return sum([vec.bfs() for vec in self.vecs], [])
