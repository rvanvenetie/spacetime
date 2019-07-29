from index_set import IndexSet
import numpy as np


class IndexedVector(object):
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
        elif isinstance(index_set, IndexSet):
            self.vector = {
                key: value
                for (key, value) in zip(sorted(index_set), values)
            }

    @classmethod
    def Zero(cls):
        """ Zero constructor. """
        return cls(index_set={})

    def __getitem__(self, key):
        if key in self.vector:
            return self.vector[key]
        return 0.0

    def __repr__(self):
        return r"IndexedVector(%s)" % self.vector

    def keys(self):
        return self.vector.keys()

    def on_level(self, l):
        return IndexedVector(
            {key: self.vector[key]
             for key in self.vector if key[0] == l})

    def __add__(self, other):
        vec = self.vector
        for key in other.vector:
            if key in vec:
                vec[key] += other.vector[key]
            else:
                vec[key] = other.vector[key]
        return IndexedVector(vec)

    def asarray(self):
        """ Slightly expensive. Mainly for testing. """
        return np.array([self[k] for k in sorted(self.keys())])

    def dot(self, index_mask, other):
        """ Dot-product; only treat indices in `index_mask` as nonzero. """
        return sum([
            self[labda] * other[labda] for labda in other.keys()
            if labda in index_mask
        ])
