from ..datastructures.functional import FunctionalInterface
from ..datastructures.tree_vector import TreeVector
from .triangulation_view import TriangulationView


class Functional(FunctionalInterface):
    """ Class can evaluate a functional on the hierarchical basis. """
    def __init__(self, singlescale_functional):
        """ Initialize the functional.

        Arguments:
            singlescale_functional: functional to apply.
        """
        super().__init__()
        self.functional = singlescale_functional

    def eval(self, Lambda):
        """ Evaluate the functional on the given functions. """
        vec_out = TreeVector.from_metaroot(Lambda.node)
        vec_out.union(Lambda)

        self.functional.triang = TriangulationView(vec_out)
        np_vec_out = self.functional.eval()
        vec_out.from_array(np_vec_out)
        return vec_out
