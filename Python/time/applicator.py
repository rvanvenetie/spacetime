from basis import Element
from sparse_vector import SparseVector


class Applicator(object):
    """ Class that can apply multiscale operators with minimal overhead.

    See also github.com/skestler/lawa-phd-skestler/LAWA-lite/
        lawa/methods/adaptive/operators/localoperators/localoperator1d.{h,tcpp}
    for some (tough to read) C++ code that implements the same operators in
    a probably more optimized fashion.
    """

    def __init__(self,
                 basis_in,
                 singlescale_operator,
                 Lambda_in=None,
                 basis_out=None,
                 Lambda_out=None):
        """ Initialize the applicator.

        Arguments:
            basis_in: A Basis object.
            singlescale_operator: LinearOperator from basis_in to basis_out.
            Lambda_in: multiscale indices to apply the multiscale operator on
                (default: the full set of indices of basis_in).
            basis_out: A Basis object (default: basis_in).
            Lambda_out: multiscale indices to compute application on
                (default: value of Lambda_in)
        """
        self.basis_in = basis_in
        self.basis_out = basis_out if basis_out else basis_in
        self.operator = singlescale_operator
        self.Lambda_in = Lambda_in if Lambda_in else basis_in.indices
        self.Lambda_out = Lambda_out if Lambda_out else self.Lambda_in

    def _initialize_elements(self):
        """ Helper function to set correct fields inside the elements. """

        def reset(elem):
            """ Reset the variables! :-) """
            elem.Lambda_in = False
            elem.Lambda_out = False
            elem.Pi_in = False
            elem.Pi_out = False
            for child in elem.children:
                reset(child)

        reset(Element.mother_element)

        for psi in self.Lambda_in:
            for elem in list(psi.support):
                elem.Lambda_in = True

        for psi in self.Lambda_out:
            for elem in list(psi.support):
                elem.Lambda_out = True

    def apply(self, vec):
        """ Apply the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        self._initialize_elements()
        e, f = self._apply_recur(
            l=1,
            Pi_in=self.Lambda_in.on_level(0),
            Pi_out=self.Lambda_out.on_level(0),
            d=vec,
            c=vec)
        return e + f

    def apply_upp(self, vec):
        """ Apply the upper part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Upper part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        self._initialize_elements()
        e, f = self._apply_upp_recur(
            l=1,
            Pi_in=self.Lambda_in.on_level(0),
            Pi_out=self.Lambda_out.on_level(0),
            d=vec,
            c=vec)
        return e + f

    def apply_low(self, vec):
        """ Apply the lower part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Lower part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        self._initialize_elements()
        f = self._apply_low_recur(
            l=1, Pi_in=self.Lambda_in.on_level(0), d=vec, c=vec)
        return f

    #  Private methods from here on out.
    def _construct_Pi_out(self, Pi_out):
        """ Singlescale indices labda st supp(phi_labda) intersects Lambda_l.

        Finds the subset Pi_B of Pi such that the support of phi_labda
        intersects some S(mu) for mu in Lambda_l.
        Necessary for computing Pi_B in apply/apply_upp/apply_low.

        Arguments:
            Pi_out: SingleLevelIndexSet of singlescale indices on level l-1.

        Returns:
            Pi_B_out: SingleLevelIndexSet of singlescale indices on level l-1.
        """
        Pi_B_out = []
        Pi_A_out = []
        for phi in Pi_out:
            # Check the support of phi on level l for wavelets psi.
            if any((child.Lambda_in for elem in phi.support
                    for child in elem.children)):
                Pi_B_out.append(phi)
            else:
                Pi_A_out.append(phi)

        return Pi_B_out, Pi_A_out

    def _construct_Pi_in(self, Pi_in, Pi_B_out):
        """ Similar to previous method, only with extra `Pi_B_out`. """
        Pi_B_in = []
        Pi_A_in = []

        # Set all the Pi_B_out boys.
        for phi in Pi_B_out:
            for elem in phi.support:
                elem.Pi_out = True

        for phi in Pi_in:
            # Check if intersects with Phi_out_Pi_B on level l - 1, or
            # with Lambda_l_out on level l.
            if any((elem.Pi_out for elem in phi.support)) or \
                 any((child.Lambda_out for elem in phi.support for child in elem.children)):
                Pi_B_in.append(phi)
            else:
                Pi_A_in.append(phi)

        # Unset all the Pi_B_out boys
        # Set all the Pi_B_out boys.
        for phi in Pi_B_out:
            for elem in phi.support:
                elem.Pi_out = False

        #TODO: The above might not be neccessary.
        return Pi_B_in, Pi_A_in

    def _apply_recur(self, l, Pi_in, Pi_out, d, c):
        """ Apply the multiscale operator on level l.

        Arguments:
            l: the current level
            Pi_in: a collection of singlescale indices on level l-1.
            Pi_out: a collection of singlescale indices on level l-1.
            d: a vector in ell_2(Pi_in).
            c: a vector in ell_2(Lambda_{l up}_in).

        Output:
            e: a vector in ell_2(Pi_bar_out)
            f: a vector in ell_2(Lambda_{l+1 up}_out).
        """
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out)
            Pi_B_in, Pi_A_in = self._construct_Pi_in(Pi_in, Pi_B_out)

            d_bar = self.basis_in.P.matvec(d, Pi_B_in,
                                           None) + self.basis_in.Q.matvec(
                                               c, Lambda_l_in, None)

            Pi_bar_in = d_bar.keys()
            Pi_bar_out = self.basis_out.P.range(
                Pi_B_out) | self.basis_out.Q.range(Lambda_l_out)

            e_bar, f_bar = self._apply_recur(l + 1, Pi_bar_in, Pi_bar_out,
                                             d_bar, c)

            e = self.operator.matvec(d, None,
                                     Pi_A_out) + self.basis_out.P.rmatvec(
                                         e_bar, None, Pi_B_out)
            f = self.basis_out.Q.rmatvec(e_bar, None, Lambda_l_out) + f_bar
            return e, f
        else:
            return SparseVector.Zero(), SparseVector.Zero()

    def _apply_upp_recur(self, l, Pi_in, Pi_out, d, c):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out)
            Pi_bar_out = self.basis_out.P.range(
                Pi_B_out) | self.basis_out.Q.range(Lambda_l_out)
            Pi_bar_in = self.basis_in.Q.range(Lambda_l_in)

            d_bar = self.basis_in.Q.matvec(c, Lambda_l_in, None)
            e_bar, f_bar = self._apply_upp_recur(l + 1, Pi_bar_in, Pi_bar_out,
                                                 d_bar, c)
            e = self.operator.matvec(d, None, Pi_out) + \
                self.basis_out.P.rmatvec(e_bar, None, Pi_B_out)
            f = self.basis_out.Q.rmatvec(e_bar, None, Lambda_l_out) + f_bar
            return e, f
        else:
            return SparseVector.Zero(), SparseVector.Zero()

    def _apply_low_recur(self, l, Pi_in, d, c):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Lambda_l_out) > 0 and len(Pi_in) + len(Lambda_l_in) > 0:
            Pi_B_in, _ = self._construct_Pi_in(Pi_in, Pi_B_out={})
            Pi_B_bar_in = self.basis_in.P.range(Pi_B_in)
            Pi_B_bar_out = self.basis_out.Q.range(Lambda_l_out)
            Pi_bar_in = self.basis_in.P.range(Pi_B_in) | self.basis_in.Q.range(
                Lambda_l_in)

            # NB: operator is applied at level `l` -- different from the rest.
            e_bar = self.operator.matvec(
                self.basis_in.P.matvec(d, Pi_B_in, None), Pi_B_bar_in,
                Pi_B_bar_out)
            d_bar = self.basis_in.P.matvec(d, Pi_B_in, None) + \
                    self.basis_in.Q.matvec(c, Lambda_l_in, None)
            f = self.basis_out.Q.rmatvec(e_bar, None,
                                         Lambda_l_out) + self._apply_low_recur(
                                             l + 1, Pi_bar_in, d_bar, c)
            return f
        else:
            return SparseVector.Zero()
