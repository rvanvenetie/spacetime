class Applicator:
    """ Class that implements a tensor product operator. """

    def __init__(self,
                 basis_in,
                 vee_in,
                 operator_time,
                 operator_space,
                 basis_out=None,
                 vee_out=None):
        """ Initialize the applicator.

        Arguments:
            basis_in: A tensor-basis input.
            vee_in: A double tree index set input corresponding to the input.
            operator_time: The operator that has to be applied to the time part.
            operator_space: The operator that has to be applied on the space part.
            basis_out: A tensor-basis output.
            vee_out: The double tree index set corresponding to the output.
        """

        self.basis_in = basis_in
        self.vee_in = vee_in
        self.operator_time = operator_time
        self.operator_space = operator_space

        if basis_out is None:
            basis_out = basis_in
            vee_out = vee_in
        self.basis_out = basis_out
        self.vee_out = basis_out

    def apply(self, vec):
        """ Apply the tensor product operator to the given vector. """

        sigma = self.sigma()
        theta = self.theta()

        # Calculate R_sigma(Id x A_2)I_vee
        proj_time_sigma = sigma.project(1)

        v = {}
        for labda in proj_time_sigma:
            labda_fiber_in = self.vee_in.fiber(2, labda)
            labda_fiber_out = self.sigma.fiber(2, labda)
            v += self.operator_space.apply(vec, labda_fiber_in,
                                           labda_fiber_out)

        # Now
