from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
from triangulation import Triangulation
from basis import support_to_interval


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


    def _triang(self):
        """ Helper function to generate a triangulation with the correct fields set. """

        class ElementExtended(Triangulation.Element):
            """ Extend the element class to hold some extra variables. """
            def __init__(self, level, node_index, parent):
                super().__init__(level, node_index, parent)

                # Add some extra variables.
                self.Lambda_in = False
                self.Lambda_out = False
                self.Pi_in = False
                self.Pi_out = False

        triang = Triangulation(ElementExtended)
        for labda in self.Lambda_in:
            for elem in triang.get_element(self.basis_in.wavelet_support(labda)):
                elem.Lambda_in = True

        for labda in self.Lambda_out:
            for elem in triang.get_element(self.basis_out.wavelet_support(labda)):
                elem.Lambda_out = True
        return triang

    def apply(self, vec):
        """ Apply the multiscale operator.

        Arguments:
            vec: an IndexedVector with indices on self.Lambda_in.

        Returns:
            self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        e, f = self._apply_recur(l=1,
                                 Pi_in=self.Lambda_in.on_level(0),
                                 Pi_out=self.Lambda_out.on_level(0),
                                 d=vec.restrict(self.Lambda_in.on_level(0)),
                                 c=vec,
                                 triang = self._triang())
        return e + f

    def apply_upp(self, vec):
        """ Apply the upper part of the multiscale operator.

        Arguments:
            vec: an IndexedVector with indices on self.Lambda_in.

        Returns:
            Upper part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        e, f = self._apply_upp_recur(l=1,
                                     Pi_in=self.Lambda_in.on_level(0),
                                     Pi_out=self.Lambda_out.on_level(0),
                                     d=vec.restrict(self.Lambda_in.on_level(0)),
                                     c=vec,
                                     triang=self._triang())
        return e + f

    def apply_low(self, vec):
        """ Apply the lower part of the multiscale operator.

        Arguments:
            vec: an IndexedVector with indices on self.Lambda_in.

        Returns:
            Lower part of self.operator(Psi_{Lambda_in})(Psi_{Lambda_out}) vec.
        """
        f = self._apply_low_recur(l=1,
                                  Pi_in=self.Lambda_in.on_level(0),
                                  d=vec.restrict(self.Lambda_in.on_level(0)),
                                  c=vec,
                                  triang=self._triang())
        return f

    #  Private methods from here on out.
    def _construct_Pi_out(self, Pi_out, triang):
        """ Singlescale indices labda st supp(phi_labda) intersects Lambda_l.

        Finds the subset Pi_B of Pi such that the support of phi_labda
        intersects some S(mu) for mu in Lambda_l.
        Necessary for computing Pi_B in apply/apply_upp/apply_low.

        Arguments:
            Pi_out: SingleLevelIndexSet of singlescale indices on level l-1.
            triang: Triangulation that has been initialized with Lambda_l_in,
                i.e.`Lambda_in` indicator has been correctly set for
                the elements inisde support of the corresponding wavelets.

        Returns:
            Pi_B_out: SingleLevelIndexSet of singlescale indices on level l-1.
        """
        Pi_B_out = []
        Pi_A_out = []

        for labda in Pi_out:
            # Generate support of Phi_out_labda on level l - 1
            support_labda = triang.get_element(self.basis_out.scaling_support(labda))

            # Check if any of the children support a wavelet on level l
            if any((child.Lambda_in for child in triang.children(support_labda))):
                Pi_B_out.append(labda)
                # TODO: possible set support_labda.Pi_out = True here.
            else:
                Pi_A_out.append(labda)

        return Pi_B_out, Pi_A_out

    def _construct_Pi_in(self, Pi_in, Pi_B_out, triang):
        """ Similar to previous method, only with extra `Pi_B_out`. """
        Pi_B_in = []
        Pi_A_in = []

        # Set all the Pi_B_out boys.
        for labda in Pi_B_out:
            for elem in triang.get_element(self.basis_out.scaling_support(labda)):
                elem.Pi_out = True

        for labda in Pi_in:
            # Generate support of Phi_in_labda on level l - 1.
            support_labda = triang.get_element(self.basis_in.scaling_support(labda))

            # Check if intersects with Phi_out_Pi_B on level l - 1, or
            # with Lambda_l_out on level l.
            if any((elem.Pi_out for elem in support_labda)) or \
               any((child.Lambda_out for child in triang.children(support_labda))):
                Pi_B_in.append(labda)
            else:
                Pi_A_in.append(labda)

        # Unset all the Pi_B_out boys
        for labda in Pi_B_out:
            for elem in triang.get_element(self.basis_out.scaling_support(labda)):
                elem.Pi_out = False
        #TODO: The above might not be neccessary.

        return Pi_B_in, Pi_A_in

    def _apply_recur(self, l, Pi_in, Pi_out, d, c, triang):
        """ Apply the multiscale operator on level l.

        Arguments:
            l: the current level
            Pi_in: a collection of singlescale indices on level l-1.
            Pi_out: a collection of singlescale indices on level l-1.
            d: a vector in ell_2(Pi_in).
            c: a vector in ell_2(Lambda_{l up}_in).
            triang: triangulation object containing the supports.

        Output:
            e: a vector in ell_2(Pi_bar_out)
            f: a vector in ell_2(Lambda_{l+1 up}_out).
        """
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out, triang)
            Pi_B_in, Pi_A_in = self._construct_Pi_in(Pi_in, Pi_B_out, triang)

            Pi_bar_out = SingleLevelIndexSet(self.basis_out.P.range(Pi_B_out) | self.basis_out.Q.range(Lambda_l_out))
            Pi_bar_in = SingleLevelIndexSet(self.basis_in.P.range(Pi_B_in) | self.basis_in.Q.range(Lambda_l_in))

            d_bar = self.basis_in.P.matvec(Pi_B_in, Pi_bar_in, d) + self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            e_bar, f_bar = self._apply_recur(l + 1, Pi_bar_in, Pi_bar_out, d_bar, c, triang)

            e = self.operator.matvec(Pi_in, Pi_A_out, d) + self.basis_out.P.rmatvec(Pi_bar_out, Pi_B_out, e_bar)
            f = self.basis_out.Q.rmatvec(Pi_bar_out, Lambda_l_out, e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_upp_recur(self, l, Pi_in, Pi_out, d, c, triang):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out, Pi_A_out = self._construct_Pi_out(Pi_out, triang)

            Pi_bar_out = SingleLevelIndexSet(self.basis_out.P.range(Pi_B_out) | self.basis_out.Q.range(Lambda_l_out))
            Pi_bar_in = SingleLevelIndexSet(self.basis_in.Q.range(Lambda_l_in))

            d_bar = self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            e_bar, f_bar = self._apply_upp_recur(l + 1, Pi_bar_in, Pi_bar_out,
                                                 d_bar, c, triang)
            e = self.operator.matvec(Pi_in, Pi_out, d) + \
                self.basis_out.P.rmatvec(Pi_bar_out, Pi_B_out, e_bar)
            f = self.basis_out.Q.rmatvec(Pi_bar_out, Lambda_l_out,
                                         e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_low_recur(self, l, Pi_in, d, c, triang):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Lambda_l_out) > 0 and len(Pi_in) + len(Lambda_l_in) > 0:
            Pi_B_in, _ = self._construct_Pi_in(Pi_in, Pi_B_out={}, triang=triang)
            Pi_bar_in = SingleLevelIndexSet(self.basis_in.P.range(Pi_B_in) | self.basis_in.Q.range(Lambda_l_in))
            Pi_B_bar_in = SingleLevelIndexSet(self.basis_in.P.range(Pi_B_in))
            Pi_B_bar_out = SingleLevelIndexSet(self.basis_out.Q.range(Lambda_l_out))

            # NB: operator is applied at level `l` -- different from the rest.
            e_bar = self.operator.matvec(
                Pi_B_bar_in, Pi_B_bar_out,
                self.basis_in.P.matvec(Pi_B_in, Pi_B_bar_in, d))
            d_bar = self.basis_in.P.matvec(Pi_B_in, Pi_bar_in, d) + \
                    self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            f = self.basis_out.Q.rmatvec(Pi_B_bar_out, Lambda_l_out,
                                         e_bar) + self._apply_low_recur(
                                             l + 1, Pi_bar_in, d_bar, c, triang)
            return f
        else:
            return IndexedVector.Zero()
