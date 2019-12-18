import numpy as np

from ..datastructures.tree_vector import TreeVector
from .basis import MultiscaleFunctions
from .sparse_vector import SparseVector


class Functional:
    """ Class that can evaluate a functional on a multiscale basis. """
    def __init__(self, singlescale_operator, basis):
        """ Initialize the functional.

        Arguments:
            singlescale_operator: Functional to evaluate in single scale.
            basis_out: A basis object on which this functional is defined.
        """
        self.operator = singlescale_operator
        self.basis_out = basis

    def eval(self, Lambda_out):
        """ Evaluate the functional on the given wavelets.

        Returns:
            SparseVector representing self.operator(Psi_{Lambda_out}).
        """
        self._initialize(Lambda_out)

        # Apply the recursive method.
        self._eval_recur(l=0, Pi_out=[])

        # Copy data back from basis into a vector, and reset variables.
        vec_out = self._finalize(Lambda_out)
        return vec_out

    def _initialize(self, Lambda):
        """ Helper function to initialize fields in datastructures. """
        assert not isinstance(Lambda, SparseVector)
        self.Lambda_out = MultiscaleFunctions(Lambda)

        # Last, update the fields inside the elements.
        for psi in self.Lambda_out:
            for elem in psi.support:
                elem.Lambda_out = True

    def _finalize(self, Lambda_out):
        """ Helper function to finalize the results.

        This also copies the data from the single trees into vec_out. """

        if isinstance(Lambda_out, MultiscaleFunctions):
            vec_out = SparseVector(Lambda_out, np.zeros(len(Lambda_out)))
            for psi in Lambda_out:
                vec_out[psi] = psi.coeff[0]
        else:

            def copy_value(nv, _):
                if not nv.is_metaroot():
                    nv.value = nv.node.coeff[0]

            vec_out = TreeVector.from_metaroot(Lambda_out.node)
            vec_out.union(Lambda_out, call_postprocess=copy_value)

        # Delete the used fields in the Element
        for psi in self.Lambda_out:
            for elem in psi.support:
                elem.Lambda_out = False

        # Delete the used fields in the input/output vector
        for psi in self.Lambda_out:
            psi.reset_coeff()

        return vec_out

    def _eval_recur(self, l, Pi_out):
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
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out)
            Pi_bar_out = self.basis_out.P.range(
                Pi_B_out) | self.basis_out.Q.range(Lambda_l_out)

            # First ensure the operator is evaluated for all lower phi.
            self._eval_recur(l + 1, Pi_bar_out)

            # Evaluate the operator for this phi.
            for phi in Pi_A_out:
                assert phi.coeff[0] == 0
                phi.coeff[0] = self.operator(phi)

            self.basis_out.P.rmatvec_inplace(None, Pi_B_out, read=0, write=0)
            self.basis_out.Q.rmatvec_inplace(None,
                                             Lambda_l_out,
                                             read=0,
                                             write=0)
            for phi in Pi_bar_out:
                phi.reset_coeff()

    def _construct_Pi_out(self, Pi_out):
        """ Singlescale indices labda st supp(phi_labda) intersects Lambda_l.

        Finds the subset Pi_B of Pi such that the support of phi_labda
        intersects some S(mu) for mu in Lambda_l, and Pi_A = Pi \ Pi_B

        Arguments:
            Pi: SingleLevelIndexSet of singlescale indices on level l-1.

        Returns:
            Pi_A: SingleLevelIndexSet of singlescale indices on level l-1.
            Pi_B: SingleLevelIndexSet of singlescale indices on level l-1.
        """
        Pi_B_out = []
        Pi_A_out = []
        for phi in Pi_out:
            # Check the support of phi on level l for wavelets psi.
            if any((child.Lambda_out for elem in phi.support
                    for child in elem.children)):
                Pi_B_out.append(phi)
            else:
                Pi_A_out.append(phi)

        return Pi_B_out, Pi_A_out
