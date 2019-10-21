import numpy as np

from ..datastructures.multi_tree_vector import MultiTreeVector
from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.tree_vector import TreeVector


class MultiTreeFunction(MultiTreeVector):
    """ Class that represents a function living on a multi tree. """
    def eval(self, coords):
        """ Evaluate in a stupid way.
        
        Arguments:
            coords: a tuple of coords, each a float or array of size (dim, N).

        Returns:
            A float, or array of shape (N,).
        """
        assert isinstance(coords, tuple)
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


class TreeFunction(MultiTreeFunction, TreeVector):
    def eval(self, coords):
        if not isinstance(coords, tuple): coords = (coords, )
        return super().eval(coords)


class DoubleTreeFunction(MultiTreeFunction, DoubleTreeVector):
    def slice_time(self, t, slice_cls=TreeFunction):
        """ Slices a double tree fn through a point in time. """
        result = slice_cls.from_metaroot(self.root.nodes[1])

        for nv in self.project(0).bfs():
            # Check if t is contained inside support of time wavelet.
            if nv.node.support_contains(t):
                result.axpy(nv.frozen_other_axis(), nv.node.eval(t))

        return result
