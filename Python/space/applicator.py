from pprint import pprint

from ..datastructures.tree_vector import MetaRootVector
from ..datastructures.tree_view import MetaRootInterface
from .triangulation_view import TriangulationView


class Applicator:
    """ Class that can apply operators on the hierarchical basis. """
    def __init__(self, singlescale_operator):
        """ Initialize the applicator.

        Arguments:
            singlescale_operator: an instance of operators.Operator.
        """
        self.operator = singlescale_operator

    def apply(self, vec_in, vec_out):
        """ Apply the multiscale operator.  """
        vec_in_nodes = vec_in.bfs()
        vec_out_nodes = vec_out.bfs()
        if len(vec_in_nodes) == 0 or len(vec_out_nodes) == 0: return
        if [n.node for n in vec_in_nodes] == [n.node for n in vec_out_nodes]:
            # This is the case where vec_in == vec_out, i.e. symmetric.
            self.operator.triang = TriangulationView(vec_in)
            np_vec_in = vec_in.to_array()
            np_vec_out = self.operator.apply(np_vec_in)
            vec_out.from_array(np_vec_out)
            return vec_out

        # This is the case where vec_in != vec_out.
        # We can handle this by enlarging vec_in and vec_out.
        def call_copy(my_node, other_node):
            if not isinstance(other_node, MetaRootInterface):
                my_node.value = other_node.value

        # If that's not the case, create an enlarged vec_in.
        vec_in_new = MetaRootVector(vec_in.node)
        vec_in_new.union(vec_in, call_postprocess=call_copy)
        vec_in_new.union(vec_out, call_postprocess=None)

        # Also create an enlarge vec_out.
        vec_out_new = MetaRootVector(vec_out.node)
        vec_out_new.union(vec_in, call_postprocess=None)
        vec_out_new.union(vec_out, call_postprocess=None)

        # Apply for these enlarged vectors.
        self.apply(vec_in_new, vec_out_new)

        # Now copy only specific parts back into vec_out.
        vec_out.union(vec_out_new,
                      call_filter=lambda _: False,
                      call_postprocess=call_copy)

        return vec_out
