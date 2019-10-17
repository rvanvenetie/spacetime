import numpy as np

from ..datastructures.multi_tree_vector import MultiTreeVector
from ..datastructures.double_tree_vector import DoubleTreeVector


class MultiTreeFunction(MultiTreeVector):
    """ Class that represents a function living on a multi tree. """
    def __init__(self, root):
        super().__init__(root=root)

    def eval(self, coords):
        """ Evaluate in a stupid way.
        
        Arguments:
            coords: a list of coords, each size (dim, N)."""
        assert len(coords) == self.root.dim
        # If the input coords are arrays, return an array, else a number.
        if any(isinstance(coord, np.ndarray) for coord in coords):
            assert all(coords[0].shape[-1] == coord.shape[-1]
                       for coord in coords)
            result = np.zeros(coords[0].shape[-1])
        else:
            result = 0.0

        for mlt_node in self.bfs():
            if mlt_node == 0.0: continue
            result += mlt_node.value * np.prod(
                [n.eval(c) for (n, c) in zip(mlt_node.nodes, coords)], axis=0)
        return result


class DoubleTreeFunction(MultiTreeFunction, DoubleTreeVector):
    pass
