import bisect
import collections.abc


class IndexSet(collections.abc.Set):
    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __contains__(self, index):
        pass


class SingleLevelIndexSet(IndexSet):
    """ Immutable set of indices on one level, either singlescale or multiscale.
    
    Would be nice if the elements in a singlescale index set would be sorted by
    lexicographical order. Probably hard to do in linear time.
    """

    def __init__(self, indices):
        """ Initialize a SLIS. Assume `indices` is a set(). """
        # Assert that all elements of this set are of the same level.
        for labda in indices:
            break
        for index in indices:
            assert index[0] == labda[0]
        self.indices = indices

        # TODO: if we assume that `indices` is given to us in lexicographical
        # ordering, we can save the list as `self.sorted` -- this also allows
        # us to loop over it and save the neighbours of every index.
        # This neighbour stuff is useful for the 3-point basis.
        self.sorted = False

    def asarray(self):
        """ Expensive. Mainly for testing. """
        if not self.sorted:
            self.sorted = sorted(self.indices)
        return self.sorted

    def neighbours(self, labda):
        """ Get the neighbours of this singlescale index.
        
        Current complexity: once O(N log N) and later O(log N).
        Goal complexity: O(1).
        TODO: this will become the bottleneck for N = 30K.
        """
        i = bisect.bisect_left(self.asarray(), labda)
        return (self.sorted[i - 1] if 0 < i < len(self.sorted) else None,
                self.sorted[i + 1] if i < len(self.sorted) - 1 else None)

    def __repr__(self):
        return r"SLIS(%s)" % self.indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def __contains__(self, index):
        return self.indices.__contains__(index)

    def __sub__(self, other):
        return SingleLevelIndexSet(self.indices - other.indices)


class MultiscaleIndexSet(IndexSet):
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
        self.sorted = False

    def __repr__(self):
        return r"MSIS(%s)" % self.indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self.indices.__iter__()

    def __next__(self):
        return self.indices.__next__()

    def __contains__(self, index):
        return self.indices.__contains__(index)

    def on_level(self, level):
        if level > self.maximum_level:
            return SingleLevelIndexSet({})
        return self.per_level[level]

    def until_level(self, level):
        """ Expensive (but linear in size) method. Mainly for testing. """
        return MultiscaleIndexSet(
            {index
             for index in self.indices if index[0] <= level})
