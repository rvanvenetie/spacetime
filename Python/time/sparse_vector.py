import collections.abc
from collections import Iterable

import numpy as np


class SparseVector(collections.abc.Mapping):
    """ A sparse vector defined on some index set.

    The representation is a dict mapping from an index to the coefficient.
    """
    def __init__(self, index_set, values=None):
        """ Initialize the vector.

        This is done with either a dictionary of index:value pairs (faster),
        or an IndexSet together with a list-like of values (slower).
        """
        if isinstance(index_set, dict):
            self.vector = index_set
        elif isinstance(index_set, Iterable) and values is not None:
            assert len(index_set) == len(values)
            self.vector = {key: value for key, value in zip(index_set, values)}
        else:
            raise TypeError('SparseVector encoutered unknown type: {}'.format(
                type(index_set)))

    @classmethod
    def Zero(cls):
        """ Zero constructor. """
        return cls(index_set={})

    def __setitem__(self, key, val):
        self.vector[key] = val

    def __getitem__(self, key):
        if key in self.vector:
            return self.vector[key]
        return 0.0

    def __iter__(self):
        return self.vector.__iter__()

    def __next__(self):
        return self.vector.__next__()

    def __len__(self):
        return len(self.vector)

    def __repr__(self):
        return r"SparseVector(%s)" % self.vector

    def restrict(self, indices):
        return SparseVector(
            {key: self.vector[key]
             for key in indices if key in self.vector})

    def __add__(self, other):
        if len(other.vector) > len(self.vector): return other + self
        vec = self.vector
        for key in other.vector:
            if key in vec:
                vec[key] += other.vector[key]
            else:
                vec[key] = other.vector[key]
        return SparseVector(vec)

    def asarray(self, keys_ordering=None):
        """ Slightly expensive. Mainly for testing. """
        if keys_ordering:
            return np.array([self[k] for k in keys_ordering])
        else:
            return np.array([self[k] for k in self.keys()])

    def deep_copy(self):
        return SparseVector(self.vector.copy())
