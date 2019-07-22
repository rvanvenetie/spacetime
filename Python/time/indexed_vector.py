import numpy as np


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

    @staticmethod
    def Zero():
        return IndexedVector(index_set={})

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

    def __add__(self, other):
        vec = self.vector
        for key in other.vector:
            if key in vec:
                vec[key] += other.vector[key]
            else:
                vec[key] = other.vector[key]
        return IndexedVector(vec)

    def asarray(self):
        return np.array([self[key] for key in sorted(self.vector.keys())],
                        dtype=float)
