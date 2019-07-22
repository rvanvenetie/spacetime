import numpy as np


class IndexSet(object):
    """ Could perhaps be a subclass of `set` """

    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def on_level(self, level):
        return IndexSet({index for index in self.indices if index[0] == level})

    def from_level(self, level):
        return IndexSet({index for index in self.indices if index[0] >= level})

    def difference(self, index_set):
        return IndexSet(self.indices.difference(index_set.indices))


class IndexedVector(object):
    """ Could perhaps be a subclass of `dict` """

    def __init__(self, index_set, values=None):
        if isinstance(index_set, dict):
            self.vector = index_set
        else:
            self.vector = {
                key: value
                for (key, value) in zip(sorted(index_set.indices), values)
            }

    def __getitem__(self, key):
        if key in self.vector:
            return self.vector[key]
        return 0.0

    def on_level(self, level):
        return IndexedVector(
            {key: self[key]
             for key in self.vector if key[0] == level})

    def from_level(self, level):
        return IndexedVector(
            {key: self[key]
             for key in self.vector if key[0] >= level})

    @classmethod
    def sum(cls, left, right):
        vec = left.vector
        for key in right.vector:
            if key in vec:
                vec[key] += right.vector[key]
            else:
                vec[key] = right.vector[key]
        return cls(vec)

    def asarray(self):
        return np.array([self[key] for key in sorted(self.vector.keys())],
                        dtype=float)
