class SingleLevelIndexSet(object):
    """ Immutable set of indices on one level, either singlescale or multiscale.
    
    Would be nice if the elements in a singlescale index set would be sorted by
    lexicographical order. Probably hard to do in linear time.
    """

    def __init__(self, indices):
        # Assert that all elements of this set are of the same level.
        for labda in indices:
            break
        for index in indices:
            assert index[0] == labda[0]
        self.indices = indices

    def __repr__(self):
        return r"SLIS(%s)" % self.indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def __sub__(self, other):
        return SingleLevelIndexSet(self.indices - other.indices)


class IndexSet(object):
    """ Immutable set of multiscale indices.

    I think Kestler does not differentiate between vectors on index sets and the
    sets themselves. Could be nice.
    """

    def __init__(self, indices):
        self.indices = indices
        self.maximum_level = max([labda[0] for labda in indices])
        per_level = [set() for _ in range(self.maximum_level + 1)]
        for labda in indices:
            per_level[labda[0]].add(labda)
        self.per_level = [SingleLevelIndexSet(S) for S in per_level]

    def __repr__(self):
        return r"IndexSet(%s)" % self.indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def on_level(self, level):
        if level > self.maximum_level:
            return SingleLevelIndexSet({})
        return self.per_level[level]

    def until_level(self, level):
        """ Expensive method. Mainly for testing. """
        return IndexSet({index for index in self.indices if index[0] <= level})
