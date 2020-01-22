from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.functional import FunctionalInterface


class TensorFunctional(FunctionalInterface):
    """ Class implements a tensor product functional on a double-tree basis. """
    def __init__(self, functional_time, functional_space):
        """ Initialize the spacetime tensor functional.

        Arguments:
            functional_time: The functional to be applied to the time axis.
            functional_space: The functional to be applied on the space axis.
        """
        self.functional_time = functional_time
        self.functional_space = functional_space

    def eval(self, Lambda_out):
        """ Evaluate the functional on the given double tree. """

        # Evalaute the functional in time and in space.
        vec_time = self.functional_time.eval(Lambda_out.project(0))
        vec_space = self.functional_space.eval(Lambda_out.project(1))

        # Store these numbers temporarily in the *real* trees
        for psi_time in vec_time.bfs():
            psi_time.node.data = psi_time.value
        for psi_space in vec_space.bfs():
            psi_space.node.data = psi_space.value

        # Double tree vector that will hold the results.
        def copy_kron(db_node, _):
            if not db_node.is_metaroot():
                db_node.value = db_node.nodes[0].data * db_node.nodes[1].data

        vec_out = Lambda_out.deep_copy(mlt_tree_cls=DoubleTreeVector,
                                       call_postprocess=copy_kron)

        # Reset the data fields.
        for psi_time in vec_time.bfs():
            psi_time.node.data = None
        for psi_space in vec_space.bfs():
            psi_space.node.data = None

        # Return the double tree.
        return vec_out
