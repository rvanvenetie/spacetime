from tree import *


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

        if self.Lambda_out.root.nodes != self.Lambda_in.root.nodes:
            raise NotImplementedError("This needs a difficult coupling!")

        # Reset element.psi_out field in preparation for Sigma.
        # This does not work when Lambda_in and Lambda_out have different bases.
        # TODO: We really need a better way to reset these fields.
        for psi_out in self.Lambda_in.project(0).bfs(include_meta_root=False):
            for elem in psi_out.node.support:
                elem.Sigma_psi_out = []
                for child in elem.children:
                    child.Sigma_psi_out = []

        for psi_out in self.Lambda_out.project(0).bfs(include_meta_root=False):
            for elem in psi_out.node.support:
                elem.Sigma_psi_out.append(psi_out.node)

        # This does not work when Lambda_in and Lambda_out have different bases.
        for psi_in in self.Lambda_in.project(1).bfs(include_meta_root=False):
            for elem in psi_in.node.support:
                elem.Theta_gamma = True

    def sigma(self):
        """ Constructs the double tree Sigma for Lambda_in and Lambda_out. """
        sigma = DoubleTree(
            self.Lambda_in.root.__class__(
                (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1])))

        # Copy the meta roots into this double tree.
        sigma.root.union(self.Lambda_in.project(0), i=0)
        sigma.root.union(self.Lambda_out.project(1), i=1)

        # In copying the 0-projection of Lambda_in into Sigma, we have copied
        # nodes that will have an empty union-of-fibers and we need to remove
        # those nodes later on :-( so let's keep track of those nodes.
        empty_labdas = []
        for psi_in_labda_0 in sigma.project(0).bfs(include_meta_root=False):
            # Get support of psi_in_labda_0 on level + 1.
            children = [
                child for elem in psi_in_labda_0.node.support
                for child in elem.children
            ]

            # Collect all fiber(1, mu) for psi_out_mu that intersect with
            # support of psi_in_labda_0, and put their union into sigma.
            is_empty = True
            for child in children:
                for mu in child.Sigma_psi_out:
                    is_empty = False
                    psi_in_labda_0.union(self.Lambda_out.fiber(1, mu))
            if is_empty:
                empty_labdas.append(psi_in_labda_0)

        # Sanity that the resulting sigma is `full`.
        for psi_in_labda in empty_labdas:
            for parent in psi_in_labda.node.parents:
                assert parent.is_full()

        # Remove the labdas that don't have a subtree.
        for psi_in_labda_0 in reversed(empty_labdas):
            psi_in_labda_0.coarsen()

        sigma.compute_fibers()
        return sigma

    def theta(self):
        theta = DoubleTree(
            self.Lambda_in.root.__class__(
                (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1])))

        theta.root.union(self.Lambda_in.project(1), i=1)
        empty_labdas = []
        for psi_in_labda_1 in theta.project(1).bfs():
            is_empty = True
            for psi_out_mu in self.Lambda_out.project(0).bfs():
                # This field does not exist.
                if any(elem.Theta_gamma for elem in psi_out_mu.support):
                    is_empty = False
                    # This operation does not exist.
                    psi_in_labda_1.insert_node(psi_out_mu)
            if is_empty:
                empty_labdas.append(psi_in_labda_1)

        for psi_in_labda_1 in reversed(empty_labdas):
            psi_in_labda_1.coarsen()

        for psi_in_labda in theta.bfs():
            assert psi_in_labda.is_full()

        theta.compute_fibers()
        return theta

    def apply(self, vec):
        """ Apply the tensor product applicator to the given vector. """

        # Calculate R_sigma(Id x A_1)I_Lambda
        v = {}
        sigma = self.sigma()
        for psi_in_labda in sigma.project(0):
            fiber_in = self.Lambda_in.fiber(1, psi_in_labda)
            fiber_out = sigma.fiber(1, psi_in_labda)
            v += self.applicator_space.apply(vec, fiber_in, fiber_out)

        # Calculate R_Lambda(L_0 x Id)I_Sigma
        w = {}
        for psi_out_labda in self.Lambda_out.project(1):
            fiber_in = sigma.fiber(0, psi_out_labda)
            fiber_out = self.Lambda_out.fiber(0, psi_out_labda)
            w += self.applicator_time.apply_low(v, fiber_in, fiber_out)

        # Calculate R_Theta(U_1 x Id)I_Lambda
        v = {}
        theta = self.theta()
        for psi_out_labda in theta.project(1):
            fiber_in = self.Lambda_in.fiber(0, psi_out_labda)
            fiber_out = theta.fiber(0, psi_out_labda)
            v += self.applicator_time.apply_upp(vec, fiber_in, fiber_out)

        # Calculate R_Lambda(id x A2)I_Theta
        for psi_out_labda in self.Lambda_out.project(0):
            fiber_in = theta.fiber(1, psi_out_labda)
            fiber_out = self.Lambda_out.fiber(1, psi_out_labda)
            w += self.applicator_space.apply(v, fiber_in, fiber_out)

        return w
