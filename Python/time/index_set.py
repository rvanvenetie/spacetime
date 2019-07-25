class IndexSet(object):
    """ Could perhaps be a subclass of `set`, or List[SingleScaleIndexSet].
    
    Would be nice if the elements in a singlescale index set would be sorted by
    lexicographical order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __repr__(self):
        return r"IndexSet(%s)" % self.indices

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

    def until_level(self, level):
        return IndexSet({index for index in self.indices if index[0] <= level})

    def difference(self, index_set):
        return IndexSet(self.indices.difference(index_set.indices))

    def maximum_level(self):
        return max([labda[0] for labda in self.indices])
