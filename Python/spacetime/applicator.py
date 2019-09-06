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
        for psi_out in self.Lambda_out.root.bfs(0):
            for elem in psi_out.nodes[0].support:
                elem.psi_out = []
        for psi_out in self.Lambda_out.root.bfs(0):
            for elem in psi_out.nodes[0].support:
                elem.psi_out.append(psi_out.nodes[0])

    def sigma(self):
        """ Constructs the double tree Sigma for Lambda_in and Lambda_out. """
        sigma_root = self.Lambda_in.root.__class__(
            (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1]))

        # Copy self.Lambda_in.project(0) into self.sigma and traverse.
        sigma_root.union(self.Lambda_in.project(0), i=0)
        # In copying the 0-projection of Lambda_in into Sigma, we have copied
        # in nodes that will have an empty union-of-fibers and we need to remove
        # those nodes later on..
        empty_labdas = []
        for psi_in_labda in sigma_root.bfs(0):
            # Get support of psi_in_labda on level + 1.
            children = [
                child for elem in psi_in_labda.nodes[0].support
                for child in elem.children
            ]

            # Collect all fiber(1, mu) for psi_out_mu that intersect with
            # support of psi_in_labda, and put their union into sigma.
            labda_empty = True
            for child in children:
                for mu in child.psi_out:
                    labda_empty = False
                    psi_in_labda.union(self.Lambda_out.fiber(1, mu), 1)
            if labda_empty:
                empty_labdas.append(psi_in_labda)

        # Sigh..
        for psi_in_labda in empty_labdas:
            for parent in psi_in_labda.nodes[0].parents:
                assert parent.is_full()
        for psi_in_labda in reversed(empty_labdas):
            psi_in_labda.coarsen()
        return DoubleTree(sigma_root)

    def theta(self):
        theta = DoubleTree(
            (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1]))
        sigma.root.union(self.Lambda_out.root, i=1)
        for psi_out_labda in theta.root.bfs(1):
            # phew...
            pass

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
