from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 FrozenDoubleNodeVector)


class Applicator:
    """ Class that implements a tensor product operator. """
    def __init__(self,
                 basis_in,
                 Lambda_in,
                 applicator_time,
                 applicator_space,
                 basis_out=None,
                 Lambda_out=None):
        """ Initialize the applicator.

        Arguments:
            basis_in: A tensor-basis input.
            Lambda_in: A double tree index set input corresponding to the input.
            applicator_time: The applicator to be applied to the time axis.
            applicator_space: The applicator to be applied on the space axis.
            basis_out: A tensor-basis output.
            Lambda_out: The double tree index set corresponding to the output.
        """
        self.basis_in = basis_in
        self.Lambda_in = Lambda_in
        self.applicator_time = applicator_time
        self.applicator_space = applicator_space

        self.basis_out = basis_out if basis_out else basis_in
        self.Lambda_out = Lambda_out if Lambda_out else Lambda_in

    def sigma(self):
        """ Constructs the double tree Sigma for Lambda_in and Lambda_out. """

        # Initialize the element.Sigma_psi_out field.
        for psi_out in self.Lambda_out.project(0).bfs():
            for elem in psi_out.node.support:
                elem.Sigma_psi_out.append(psi_out.node)

        # Sigma is actually an empty vector.
        sigma_root = DoubleNodeVector(nodes=(self.Lambda_in.root.nodes[0],
                                             self.Lambda_out.root.nodes[1]),
                                      value=0)
        sigma = DoubleTree(sigma_root, frozen_dbl_cls=FrozenDoubleNodeVector)

        # Insert the `time single tree` x `space meta root` and
        # the `time meta root` x `space single tree` into Sigma.
        # This will copy all nodes in time and space into the tree, also the
        # ones without any subtree. This won't be a big issue, since either
        # coordinate will always be of type MetaRoot.
        sigma.project(0).union(self.Lambda_in.project(0))
        sigma.project(1).union(self.Lambda_out.project(1))

        for psi_in_labda_0 in sigma.project(0).bfs():
            # Get support of psi_in_labda_0 on level + 1.
            children = [
                child for elem in psi_in_labda_0.node.support
                for child in elem.children
            ]

            # Collect all fiber(1, mu) for psi_out_mu that intersect with
            # support of psi_in_labda_0, use set to deduplicate.
            mu_out = set(mu for child in children
                         for mu in child.Sigma_psi_out)

            # Put their union into sigma.
            for mu in mu_out:
                psi_in_labda_0.frozen_other_axis().union(
                    self.Lambda_out.fiber(1, mu))

        sigma.compute_fibers()

        # Reset element.psi_out field for futher usage.
        for psi_out in self.Lambda_out.project(0).bfs():
            for elem in psi_out.node.support:
                elem.Sigma_psi_out = []
        return sigma

    def theta(self):
        # Theta is actually an empty vector.
        theta_root = DoubleNodeVector(nodes=(self.Lambda_out.root.nodes[0],
                                             self.Lambda_in.root.nodes[1]),
                                      value=0)
        theta = DoubleTree(theta_root, frozen_dbl_cls=FrozenDoubleNodeVector)

        # Load the metaroot axes.
        theta.project(0).union(self.Lambda_out.project(0))
        theta.project(1).union(self.Lambda_in.project(1))

        # Loop over the space axis.
        for psi_in_labda_1 in theta.project(1).bfs():
            # Get the fiber of psi_in_labda in the original tree.
            fiber_labda_0 = self.Lambda_in.fiber(0, psi_in_labda_1)

            # Set the support of fn's in fiber_labda_0 to True.
            for psi_in_labda_0 in fiber_labda_0.bfs():
                for elem in psi_in_labda_0.node.support:
                    elem.Theta_psi_in = True

            # Now add all nodes inside Lambda_out.project(0) that have
            # intersecting support with fiber_labda_0 (and lie on the same lvl)
            # This works by using a callback that returns whether labda_out,
            # a node from the in the Labda_out.project(0) tree, has intersecting
            # support with a labda_in, a node from fiber_labda_0.
            call_filter = lambda psi_out_labda_0: any(
                elem.Theta_psi_in for elem in psi_out_labda_0.support)
            psi_in_labda_1.frozen_other_axis().union(
                self.Lambda_out.project(0), call_filter=call_filter)

            # Reset the support of fn's in fiber_labda_0.
            for psi_in_labda_0 in fiber_labda_0.bfs():
                for elem in psi_in_labda_0.node.support:
                    elem.Theta_psi_in = False

        theta.compute_fibers()
        return theta

    def apply(self, vec_in, vec_out):
        """ Apply the tensor product applicator to the given vector. """
        # Assert that vec_in and vec_out are defined on Lambda_in and Lambda_out
        assert all(n1.nodes == n2.nodes
                   for n1, n2 in zip(vec_in.bfs(), self.Lambda_in.bfs()))
        assert all(n1.nodes == n2.nodes
                   for n1, n2 in zip(vec_out.bfs(), self.Lambda_out.bfs()))

        # Assert that the output vector is empty
        assert isinstance(vec_out.root, DoubleNodeVector)
        assert all(db_node.value == 0
                   for db_node in vec_out.bfs(include_meta_root=True))

        # Calculate R_sigma(Id x A_1)I_Lambda
        sigma = self.sigma()
        assert isinstance(sigma.root, DoubleNodeVector)
        for psi_in_labda in sigma.project(0).bfs():
            fiber_in = vec_in.fiber(1, psi_in_labda)
            fiber_out = sigma.fiber(1, psi_in_labda)
            self.applicator_space.apply(fiber_in, fiber_out)

        # Calculate R_Lambda(L_0 x Id)I_Sigma
        for psi_out_labda in vec_out.project(1).bfs():
            fiber_in = sigma.fiber(0, psi_out_labda)
            fiber_out = vec_out.fiber(0, psi_out_labda)
            self.applicator_time.apply_low(fiber_in, fiber_out)

        # Calculate R_Theta(U_1 x Id)I_Lambda
        theta = self.theta()
        for psi_out_labda in theta.project(1).bfs():
            fiber_in = vec_in.fiber(0, psi_out_labda)
            fiber_out = theta.fiber(0, psi_out_labda)
            self.applicator_time.apply_upp(fiber_in, fiber_out)

        # Calculate R_Lambda(id x A2)I_Theta
        for psi_out_labda in vec_out.project(0).bfs():
            fiber_in = theta.fiber(1, psi_out_labda)
            fiber_out = vec_out.fiber(1, psi_out_labda)
            self.applicator_space.apply(fiber_in, fiber_out)
