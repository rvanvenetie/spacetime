from ..datastructures.tree_vector import TreeVector
from .triangulation_view import TriangulationView


class Functional:
    """ Class can evaluate a functional on the hierarchical basis. """
    def __init__(self, singlescale_operator):
        """ Initialize the functional.

        Arguments:
            singlescale_operator: an instance of operators.Operator.
        """
        super().__init__()
        self.operator = singlescale_operator

    def eval(self, Lambda):
        """ Evaluate the functional on the given functions. """
        vec_out = TreeVector.from_metaroot(Lambda.node)
        vec_out.union(Lambda)

        self.operator.triang = TriangulationView(vec_out)
        np_vec_out = self.operator.apply()
        vec_out.from_array(np_vec_out)
        return vec_out
