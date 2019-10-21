import numpy as np

from ..datastructures.applicator import ApplicatorInterface
from ..datastructures.tree_view import NodeViewInterface
from .basis import MultiscaleFunctions
from .sparse_vector import SparseVector


class Applicator(ApplicatorInterface):
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
        super().__init__()
        self.operator = singlescale_operator
        self.basis_in = basis_in
        self.basis_out = basis_out if basis_out else basis_in

    def _initialize(self, vec_in, vec_out):
        """ Helper function to initialize fields in datastructures. """

        # Sanity check that we start with an empty vector
        if isinstance(vec_out, NodeViewInterface):
            for nv in vec_out.bfs():
                assert nv.value == 0
        else:
            for psi, value in vec_out.items():
                assert value == 0

        self.Lambda_in = MultiscaleFunctions(vec_in)
        self.Lambda_out = MultiscaleFunctions(vec_out)

        # Store the vector inside the wavelet tree.
        if isinstance(vec_in, NodeViewInterface):
            for nv in vec_in.bfs():
                assert nv.node.coeff[0] == 0
                nv.node.coeff[0] = nv.value
        else:
            for psi, value in vec_in.items():
                assert psi.coeff[0] == 0
                psi.coeff[0] = value

        # Last, update the fields inside the elements.
        for psi in self.Lambda_in:
            for elem in psi.support:
                elem.Lambda_in = True

        for psi in self.Lambda_out:
            for elem in psi.support:
                elem.Lambda_out = True

    def _finalize(self, vec_in, vec_out):
        """ Helper function to finalize the results. 

        This also copies the data from the single trees into vec_out. """

        # Copy result into vec_out
        if isinstance(vec_out, NodeViewInterface):
            for nv in vec_out.bfs():
                nv.value = nv.node.coeff[1]
        else:
            # TODO: This should be removed
            for psi in self.Lambda_out:
                vec_out[psi] = psi.coeff[1]

        # Delete the used fields in the Element
        for psi in self.Lambda_in:
            for elem in psi.support:
                elem.Lambda_in = False
        for psi in self.Lambda_out:
            for elem in psi.support:
                elem.Lambda_out = False

        # Delete the used fields in the input/output vector
        for psi in self.Lambda_out:
            psi.reset_coeff()
        for psi in self.Lambda_in:
            psi.reset_coeff()

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

        # Copy data back from basis into a vector, and reset variables.
        self._finalize(vec_in, vec_out)
        return vec_out

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

        self._finalize(vec_in, vec_out)
        return vec_out

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

        self._finalize(vec_in, vec_out)
        return vec_out

    def transpose(self):
        """ Returns an applicator for the tranpose of this bilinear form. """
        return Applicator(self.operator.transpose(),
                          basis_in=self.basis_out,
                          basis_out=self.basis_in)

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
            for phi in Pi_B_bar_out:
                phi.reset_coeff()

    def to_matrix(self, Lambda_in, Lambda_out):
        """ Returns the dense matrix. Debug function. O(n^2). """
        if isinstance(Lambda_in, NodeViewInterface):
            Lambda_in = [nv.node for nv in Lambda_in.bfs()]
        if isinstance(Lambda_out, NodeViewInterface):
            Lambda_out = [nv.node for nv in Lambda_out.bfs()]

        n, m = len(Lambda_out), len(Lambda_in)
        result = np.zeros((n, m))
        for i, psi in enumerate(Lambda_in):
            vec_in = SparseVector(Lambda_in, np.zeros(m))
            vec_in[psi] = 1.0

            vec_out = SparseVector(Lambda_out, np.zeros(n))
            self.apply(vec_in, vec_out)
            for j, phi in enumerate(Lambda_out):
                result[j, i] = vec_out[phi]
        return result
