from ..datastructures.tree_view import NodeViewInterface
from .basis import MultiscaleFunctions, mother_element
from .sparse_vector import SparseVector


class Applicator(object):
    """ Class that can apply multiscale operators with minimal overhead.

    See also github.com/skestler/lawa-phd-skestler/LAWA-lite/
        lawa/methods/adaptive/operators/localoperators/localoperator1d.{h,tcpp}
    for some (tough to read) C++ code that implements the same operators in
    a probably more optimized fashion.
    """
    def __init__(self, singlescale_operator, basis_in, basis_out=None):
        """ Initialize the applicator.

        Arguments:
            singlescale_operator: LinearOperator from basis_in to basis_out.
            basis_in: A Basis object.
            basis_out: A Basis object (default: basis_in).
        """
        self.operator = singlescale_operator
        self.basis_in = basis_in
        self.basis_out = basis_out if basis_out else basis_in

    def _copy_result_into_vec_out(self, vec_out):
        # TODO: This is a legacy function. Should be removed.
        if isinstance(vec_out, NodeViewInterface):
            for nv in vec_out.bfs():
                nv.value = nv.node.coeff[1]
        else:
            for psi in self.Lambda_out:
                vec_out[psi] = psi.coeff[1]
        return vec_out

    def _initialize(self, vec_in, vec_out):
        """ Helper function to initialize fields in datastructures. """

        self.Lambda_in = MultiscaleFunctions(vec_in)
        self.Lambda_out = MultiscaleFunctions(vec_out)

        # Reset the output vector.
        for psi in self.Lambda_out:
            psi.reset_coeff()

        # First, store the vector inside the wavelet.
        # TODO: This should be removed.
        for psi, value in vec_in.items():
            psi.reset_coeff()
            psi.coeff[0] = value

        # Second, reset data inside the `elements`.
        # TODO: This is non-linear, fix this.
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

    def apply(self, vec_in, vec_out=None):
        """ Apply the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        if vec_out is None: vec_out = SparseVector({psi: 0 for psi in vec_in})
        self._initialize(vec_in, vec_out)

        # Apply the recursive method.
        self._apply_recur(l=0, Pi_in=[], Pi_out=[])

        # Copy data back from basis into a vector.
        return self._copy_result_into_vec_out(vec_out)

    def apply_upp(self, vec_in, vec_out=None):
        """ Apply the upper part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Upper part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        if vec_out is None: vec_out = SparseVector({psi: 0 for psi in vec_in})
        self._initialize(vec_in, vec_out)

        self._apply_upp_recur(l=0, Pi_in=[], Pi_out=[])
        return self._copy_result_into_vec_out(vec_out)

    def apply_low(self, vec_in, vec_out=None):
        """ Apply the lower part of the multiscale operator.

        Arguments:
            vec: an SparseVector with indices on self.Lambda_in.

        Returns:
            Lower part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        if vec_out is None: vec_out = SparseVector({psi: 0 for psi in vec_in})
        self._initialize(vec_in, vec_out)

        self._apply_low_recur(l=0, Pi_in=[])
        return self._copy_result_into_vec_out(vec_out)

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
