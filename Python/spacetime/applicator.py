class Applicator:
    """ Class that implements a tensor product operator. """

    def __init__(self,
                 basis_in,
                 Lambda_in,
                 operator_time,
                 operator_space,
                 basis_out=None,
                 Lambda_out=None):
        """ Initialize the applicator.

        Arguments:
            basis_in: A tensor-basis input.
            Lambda_in: A double tree index set input corresponding to the input.
            operator_time: The operator to be applied to the time axis.
            operator_space: The operator to be applied on the space axis.
            basis_out: A tensor-basis output.
            Lambda_out: The double tree index set corresponding to the output.
        """
        self.basis_in = basis_in
        self.Lambda_in = Lambda_in
        self.operator_time = operator_time
        self.operator_space = operator_space

        if basis_out is None:
            basis_out = basis_in
            Lambda_out = Lambda_in
        self.basis_out = basis_out
        self.Lambda_out = Lambda_out

    def sigma(self):
        """ Returns the double tree Sigma for Lambda_in and Lambda_out. """
        result = {}
        for psi_in_labda in self.Lambda_in.project(1):
            # Get support of psi_in_labda on level + 1.
            children = [
                child for elem in psi_in_labda.support
                for child in elem.children
            ]

            # Collect all fiber(2, mu) for psi_out_mu that
            # intersect with support of psi_in_labda.
            space_out = set([
                self.Lambda_out.fiber(2, mu) for child in children
                for mu in child.psi_out
            ])

            result[psi_in_labda] = space_out
        return result

    def apply(self, vec):
        """ Apply the tensor product operator to the given vector. """

        sigma = self.sigma()
        theta = self.theta()

        # Calculate R_sigma(Id x A_2)I_Lambda
        proj_time_sigma = sigma.project(1)

        v = {}
        for psi_in_labda in proj_time_sigma:
            labda_fiber_in = self.Lambda_in.fiber(2, psi_in_labda)
            labda_fiber_out = self.sigma.fiber(2, psi_in_labda)
            v += self.operator_space.apply(vec, labda_fiber_in,
                                           labda_fiber_out)

        # Now
