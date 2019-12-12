import numpy as np

from ..datastructures.applicator import ApplicatorInterface
from ..datastructures.tree_vector import TreeVector
from .triangulation_view import TriangulationView


class Applicator(ApplicatorInterface):
    """ Class that can apply operators on the hierarchical basis. """
    def __init__(self, singlescale_operator):
        """ Initialize the applicator.

        Arguments:
            singlescale_operator: an instance of operators.Operator.
        """
        super().__init__()
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
            if not my_node.is_metaroot():
                my_node.value = other_node.value

        # If that's not the case, create an enlarged vec_in.
        vec_in_new = TreeVector.from_metaroot(vec_in.node)
        vec_in_new.union(vec_in, call_postprocess=call_copy)
        vec_in_new.union(vec_out, call_postprocess=None)

        # Also create an enlarge vec_out.
        vec_out_new = TreeVector.from_metaroot(vec_in.node)
        vec_out_new.union(vec_in, call_postprocess=None)
        vec_out_new.union(vec_out, call_postprocess=None)

        # Apply for these enlarged vectors.
        self.apply(vec_in_new, vec_out_new)

        # Now copy only specific parts back into vec_out, mark these nodes.
        vec_out.union(vec_out_new,
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


class BlockDiagonalApplicator(ApplicatorInterface):
    """ Class that implements R_Lambda (Id x A) I_Lambda. """
    def __init__(self, Lambda, applicator_space):
        super().__init__(Lambda_in=Lambda, Lambda_out=Lambda)
        self.Lambda_in.compute_fibers()
        self.applicator_space = applicator_space
        self.vec_out = self.Lambda_out.deep_copy(mlt_tree_cls=DoubleTreeVector)

    def apply(self, vec):
        self.vec_out.reset()

        for psi_in_lambda in self.Lambda_out.project(0).bfs():
            fiber_in = vec.fiber(1, psi_in_lambda)
            fiber_out = self.vec_out.fiber(1, psi_in_lambda)
            self.applicator_space.apply(fiber_in, fiber_out)

        return self.vec_out

    def transpose(self):
        """ Transposes this spacetime bilinear formulation. """
        return TimeIdentityApplicator(
            Lambda=self.Lambda,
            applicator_space=self.applicator_space.transpose())
