import numpy as np

from .double_tree_view import (DoubleNodeView, DoubleTreeView,
                               FrozenDoubleNodeView)
from .multi_tree_vector import (MultiNodeVector, MultiNodeVectorInterface,
                                MultiTreeVector)


class DoubleNodeVector(MultiNodeVector, DoubleNodeView):
    pass


class FrozenDoubleNodeVector(MultiNodeVectorInterface, FrozenDoubleNodeView):
    @property
    def value(self):
        return self.dbl_node.value

    @value.setter
    def value(self, other):
        self.dbl_node.value = other

    def to_array(self):
        """ Transforms a double tree vector to a numpy vector.  """
        nodes = self.bfs()
        result = np.array([node.value for node in nodes], dtype=float)
        return result
        return np.array([node.value for node in self.bfs()], dtype=float)

    def from_array(self, array):
        """ Loads the values from the array in BFS-order into the multi treevector. """
        nodes = self.bfs()
        assert len(nodes) == len(array)
        for idx, node in enumerate(nodes):
            node.value = array[idx]
        return self


class DoubleTreeVector(MultiTreeVector, DoubleTreeView):
    def __init__(self, root, frozen_dbl_cls=FrozenDoubleNodeVector):
        if isinstance(root, (tuple, list)): root = DoubleNodeVector(root)
        super().__init__(root=root, frozen_dbl_cls=frozen_dbl_cls)

    def to_array(self):
        """ Transforms a double tree vector to a numpy vector. 
        
        This overrides the default behaviour, but this is consistent with the
        np.kron ordering of dofs.
        """
        return np.array([
            psi_1.value for psi_0 in self.project(0).bfs()
            for psi_1 in psi_0.frozen_other_axis().bfs()
        ],
                        dtype=float)

    def from_array(self, array):
        """ Loads the values from an numpy array into this double vector. """
        i = 0
        for psi_0 in self.project(0).bfs():
            for psi_1 in psi_0.frozen_other_axis().bfs():
                psi_1.value = array[i]
                i += 1
