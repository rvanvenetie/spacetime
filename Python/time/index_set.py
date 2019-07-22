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
