from index_set import IndexSet
import numpy as np
import collections.abc
from collections import Iterable


class IndexedVector(collections.abc.Mapping):
    """ A vector defined on an index set.

    If it is a Tuple[IndexSet, scipy.sparse.*], then methods like `on_level()`
    could be implemented efficiently, I think.
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
            self.vector = {key : value for key, value in zip(index_set, values)}
        else:
            raise TypeError('IndexedVector encoutered unknown type: {}'.format(type(index_set)))

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
        return self.vector.__iter__()

    def __len__(self):
        return len(self.vector)

    def __repr__(self):
        return r"IndexedVector(%s)" % self.vector

    def restrict(self, indices):
        return IndexedVector(
            {key: self.vector[key] for key in indices if key in self.vector})

    def __add__(self, other):
        vec = self.vector
        for key in other.vector:
            if key in vec:
                vec[key] += other.vector[key]
            else:
                vec[key] = other.vector[key]
        return IndexedVector(vec)

    def asarray(self, keys_ordering=None):
        """ Slightly expensive. Mainly for testing. """
        if keys_ordering:
            return np.array([self[k] for k in keys_ordering])
        else:
            return np.array([self[k] for k in self.keys()])

    def dot(self, index_mask, other):
        """ Dot-product; only treat indices in `index_mask` as nonzero. """
        if not isinstance(index_mask, collections.abc.Set):
            raise TypeError('For vec.dot to be efficient, the index_mask must be a set type. We got {}.'.format(type(index_mask)))
        return sum(self.vector[labda] * other[labda] for labda in other.keys()
                   if labda in index_mask)
