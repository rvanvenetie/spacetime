from dag import union


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

        if basis_out is None:
            basis_out = basis_in
            Lambda_out = Lambda_in
        self.basis_out = basis_out
        self.Lambda_out = Lambda_out

        self.construct_sigma()

    def sigma(self):
        """ Constructs the double tree Sigma for Lambda_in and Lambda_out. """
        sigma = DoubleTree(
            (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1]))

        # Copy self.Lambda_in.project(0) into self.sigma and traverse.
        sigma.root.union_from(self.Lambda_in.root, i=0)
        for psi_in_labda in self.sigma.root.bfs(0):
            # Get support of psi_in_labda on level + 1.
            children = [
                child for elem in psi_in_labda.support
                for child in elem.children
            ]

            # Collect all fiber(1, mu) for psi_out_mu that intersect with
            # support of psi_in_labda, and put their union into sigma.
            for child in children:
                for mu in child.psi_out:
                    psi_in_labda.union_from(self.Lambda_out.fiber(1, mu), 1)
        return sigma

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

        # Now theta
        theta = self.theta()
