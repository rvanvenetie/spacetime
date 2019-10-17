import numpy as np

from ..datastructures.multi_tree_vector import MultiTreeVector


class MultiTreeFunction(MultiTreeVector):
    """ Class that represents a function living on a multi tree. """
    def __init__(self, root):
        super().__init__(root=root)

    def eval(self, args, deriv=(False, False)):
        """ Evaluate in a stupid way. """
        assert x.shape[0] == 2
        if isinstance(t, np.ndarray):
            assert t.shape[0] == x.shape[1]
            result = np.zeros(t.shape) if not deriv[1] else np.zeros(x.shape)
        else:
            result = 0.0 if not deriv[1] else np.zeros(2)

        for node in self.bfs():
            result += node.value * node.nodes[0].eval(
                t, deriv[0]) * node.nodes[1].eval(x, deriv[1])
        return result
