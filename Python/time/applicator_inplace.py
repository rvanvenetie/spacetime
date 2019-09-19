from .basis import Element, mother_element
from .sparse_vector import SparseVector


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

    def _initialize(self, vec):
        """ Helper function to initialize fields in datastructures. """

        # First, store the vector inside the wavelet.
        # TODO: This should be removed.
        for psi, value in vec.items():
            psi.coeff[0] = value

        # Second, reset data inside the `elements`.
        def reset(elem):
            """ Reset the variables! :-) """
            elem.Lambda_in = False
            elem.Lambda_out = False
            elem.Pi_in = False
            elem.Pi_out = False
            for child in elem.children:
                reset(child)

        # Recursively resets all elements.
        reset(mother_element)

        # Last, update the fields inside the elements.
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
        self._initialize(vec)

        # Apply the recursive method.
        self._apply_recur(l=1,
                          Pi_in=self.Lambda_in.on_level(0),
                          Pi_out=self.Lambda_out.on_level(0))

        # Copy data back from basis into a vector.
        return SparseVector({psi: psi.coeff[1] for psi in self.Lambda_out})

    def apply_upp(self, vec):
        """ Apply the upper part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Upper part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        self._initialize(vec)

        self._apply_upp_recur(l=1,
                              Pi_in=self.Lambda_in.on_level(0),
                              Pi_out=self.Lambda_out.on_level(0))

        return SparseVector({psi: psi.coeff[1] for psi in self.Lambda_out})

    def apply_low(self, vec):
        """ Apply the lower part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Lower part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        self._initialize(vec)

        self._apply_low_recur(l=1, Pi_in=self.Lambda_in.on_level(0))
        return SparseVector({psi: psi.coeff[1] for psi in self.Lambda_out})

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
            if any((elem.Pi_out for elem in phi.support)) or any(
                (c.Lambda_out for e in phi.support for c in e.children)):
                Pi_B_in.append(phi)
            else:
                Pi_A_in.append(phi)

        # Unset all the Pi_B_out boys.
        # TODO: Might not be neccessary.
        for phi in Pi_B_out:
            for elem in phi.support:
                elem.Pi_out = False

        return Pi_B_in, Pi_A_in

    def _apply_recur(self, l, Pi_in, Pi_out):
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

            self.basis_in.P.matvec_inplace(Pi_B_in, None, read=0, write=0)
            self.basis_in.Q.matvec_inplace(Lambda_l_in, None, read=0, write=0)

            Pi_bar_in = self.basis_in.P.range(Pi_B_in) | self.basis_in.Q.range(
                Lambda_l_in)
            Pi_bar_out = self.basis_out.P.range(
                Pi_B_out) | self.basis_out.Q.range(Lambda_l_out)

            self._apply_recur(l + 1, Pi_bar_in, Pi_bar_out)

            self.operator.matvec_inplace(None, Pi_A_out, read=0, write=1)
            self.basis_out.P.rmatvec_inplace(None, Pi_B_out, read=1, write=1)
            self.basis_out.Q.rmatvec_inplace(None,
                                             Lambda_l_out,
                                             read=1,
                                             write=1)
            for phi in Pi_bar_in:
                phi.reset_coeff()
            for phi in Pi_bar_out:
                phi.reset_coeff()

    def _apply_upp_recur(self, l, Pi_in, Pi_out):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out)
            Pi_bar_out = self.basis_out.P.range(
                Pi_B_out) | self.basis_out.Q.range(Lambda_l_out)
            Pi_bar_in = self.basis_in.Q.range(Lambda_l_in)

            self.basis_in.Q.matvec_inplace(Lambda_l_in, None, read=0, write=0)
            self._apply_upp_recur(l + 1, Pi_bar_in, Pi_bar_out)
            self.operator.matvec_inplace(None, Pi_out, read=0, write=1)
            self.basis_out.P.rmatvec_inplace(None, Pi_B_out, read=1, write=1)
            self.basis_out.Q.rmatvec_inplace(None,
                                             Lambda_l_out,
                                             read=1,
                                             write=1)

            for phi in Pi_bar_in:
                phi.reset_coeff()
            for phi in Pi_bar_out:
                phi.reset_coeff()

    def _apply_low_recur(self, l, Pi_in):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Lambda_l_out) > 0 and len(Pi_in) + len(Lambda_l_in) > 0:
            Pi_B_in, _ = self._construct_Pi_in(Pi_in, Pi_B_out={})
            Pi_B_bar_in = self.basis_in.P.range(Pi_B_in)
            Pi_B_bar_out = self.basis_out.Q.range(Lambda_l_out)
            Pi_bar_in = self.basis_in.P.range(Pi_B_in) | self.basis_in.Q.range(
                Lambda_l_in)

            self.basis_in.P.matvec_inplace(Pi_B_in, None, read=0, write=0)
            # NB: operator is applied at level `l` -- different from the rest.
            self.operator.matvec_inplace(None, Pi_B_bar_out, read=0, write=1)
            self.basis_in.Q.matvec_inplace(Lambda_l_in, None, read=0, write=0)

            self.basis_out.Q.rmatvec_inplace(None,
                                             Lambda_l_out,
                                             read=1,
                                             write=1)
            self._apply_low_recur(l + 1, Pi_bar_in)

            for phi in Pi_bar_in:
                phi.reset_coeff()
