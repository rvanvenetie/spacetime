import numpy as np

from ..datastructures.applicator import ApplicatorInterface
from ..datastructures.tree_vector import TreeVector
from .triangulation_view import TriangulationView


class Applicator(ApplicatorInterface):
    """ Class that can apply operators on the hierarchical basis. """
    def __init__(self, singlescale_operator, use_cache=False):
        """ Initialize the applicator.

        Arguments:
            singlescale_operator: an instance of operators.Operator.
            use_cache: this caches TriangulationView objects, use only if you
              evaluate this applicator multiple times.
        """
        super().__init__()
        self.operator = singlescale_operator

        self.use_cache = use_cache
        self.triang_view_cache = {}

    def apply(self, vec_in, vec_out, **kwargs):
        """ Apply the multiscale operator. """
        vec_in_nodes = tuple(n.node for n in vec_in.bfs())
        if vec_out is vec_in:
            vec_out_nodes = vec_in_nodes
        else:
            vec_out_nodes = tuple(n.node for n in vec_out.bfs())
        if len(vec_in_nodes) == 0 or len(vec_out_nodes) == 0: return
        if vec_in_nodes == vec_out_nodes:
            # This is the case where vec_in == vec_out, i.e. symmetric.

            # Create a triangulation view object.
            if not self.use_cache:
                self.operator.triang = TriangulationView(vec_in)
            else:
                # Cache triangulation view objects..
                if vec_in_nodes not in self.triang_view_cache:
                    self.triang_view_cache[vec_in_nodes] = TriangulationView(
                        vec_in)
                self.operator.triang = self.triang_view_cache[vec_in_nodes]

            np_vec_in = vec_in.to_array()
            np_vec_out = self.operator.apply(np_vec_in, **kwargs)
            vec_out.from_array(np_vec_out)
            return vec_out

        # This is the case where vec_in != vec_out.
        # We can handle this by enlarging vec_in and vec_out.
        def call_copy(my_node, other_node):
            my_node.value = other_node.value

        # If that's not the case, create an enlarged vec_in.
        vec_in_new = TreeVector.from_metaroot(vec_in.node)
        vec_in_new.union(vec_in, call_postprocess=call_copy)
        vec_in_new.union(vec_out, call_postprocess=None)

        # Apply the operator, and store result in the same vector.
        self.apply(vec_in_new, vec_in_new)

        # Now copy only specific parts back into vec_out, mark these nodes.
        vec_out.union(vec_in_new,
                      call_filter=lambda _: False,
                      call_postprocess=call_copy)

        return vec_out

    def transpose(self):
        """ All of the space applicators are self-adjoint. """
        return self

    def to_matrix(self, Lambda_in, Lambda_out):
        """ Returns the dense matrix. Debug function. O(n^2). """
        nodes_in = Lambda_in.bfs()
        nodes_out = Lambda_out.bfs()

        n, m = len(nodes_out), len(nodes_in)
        result = np.zeros((n, m))
        for i, psi in enumerate(nodes_in):
            # Create vector with a 1 for psi
            vec_in = TreeVector.from_metaroot(Lambda_in.node)
            vec_in.union(Lambda_in)
            for n in vec_in.bfs():
                if n.node == psi.node:
                    n.value = 1
                    break
            assert sum(n.value for n in vec_in.bfs()) == 1

            vec_out = TreeVector.from_metaroot(Lambda_out.node)
            vec_out.union(Lambda_out)
            assert sum(n.value for n in vec_out.bfs()) == 0
            self.apply(vec_in, vec_out)
            for j, phi in enumerate(vec_out.bfs()):
                result[j, i] = phi.value
        return result
