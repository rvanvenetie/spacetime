from index_set import MultiscaleIndexSet, SingleLevelIndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet


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
        self.Lambda_out = Lambda_out if Lambda_out else Lambda_in

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
                                 d=vec.on_level(0),
                                 c=vec)
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
                                     d=vec.on_level(0),
                                     c=vec)
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
                                  d=vec.on_level(0),
                                  c=vec)
        return f

    #  Private methods from here on out.
    def _construct_Pi_B_out(self, Pi_out, Lambda_l_in):
        """ Singlescale indices labda st supp(phi_labda) intersects Lambda_l.
        
        Finds the subset Pi_B of Pi such that the support of phi_labda
        intersects some S(mu) for mu in Lambda_l.
        Necessary for computing Pi_B in apply/apply_upp/apply_low.

        Goal complexity: O(|Pi_B|).
        Current complexity: O(|PiBin| * |LambdaL| log[|LabdaL|]).
        TODO: make faster. it could be faster/easier to compute Pi_A?

        Arguments:
            Pi_out: SingleLevelIndexSet of singlescale indices on level l-1.
            Lambda_l_in: SingleLevelIndexSet of multiscale indices on level l.

        Returns:
            Pi_B_out: SingleLevelIndexSet of singlescale indices on level l-1.
        """
        ivs = IntervalSet(
            [self.basis_in.wavelet_nbrhood(mu) for mu in Lambda_l_in])
        return SingleLevelIndexSet({
            index
            for index in Pi_out
            if ivs.intersects(self.basis_out.scaling_support(index))
        })

    def _construct_Pi_B_in(self, Pi_in, Lambda_l_out, Pi_B_out):
        """ Similar to previous method, only with extra `Pi_B_out`.

        Goal complexity: O(|Pi_B_in|).
        Current complexity: O(PiBin * (PiBout + LambdaL) log[PiBout + LabdaL]).
        """
        ivs = IntervalSet(
            [self.basis_out.wavelet_nbrhood(mu) for mu in Lambda_l_out] +
            [self.basis_in.scaling_support(mu) for mu in Pi_B_out])
        return SingleLevelIndexSet({
            index
            for index in Pi_in
            if ivs.intersects(self.basis_in.scaling_support(index))
        })

    def _smallest_superset(self, l, basis, Pi_B, Lambda_l):
        """ Find Pi_bar, the smallest index set covering Pi_B and Lambda_l.

        span Phi_{Pi_B} cup span Psi_{Lambda_l} subset span Phi_{Pi_bar}.

        Goal complexity: O(|Pi_bar|).
        Current compl: O(|Pi_bar|*((Pi_B + Lambda_l) log[Pi_B + Lambda_L])).
        TODO: make this quicker.

        Arguments:
            l: the current level.
            basis: the basis to work on.
            Pi_B: singlescale indices on level l-1.
            Lambda_l: multiscale indices on level l.

        Returns:
            Pi_bar: singlescale indices on level l.
        """
        ivs = IntervalSet([basis.scaling_support(index) for index in Pi_B] +
                          [basis.wavelet_support(index) for index in Lambda_l])
        return SingleLevelIndexSet({
            index
            for index in basis.scaling_indices_on_level(l)
            if ivs.covers(basis.scaling_support(index))
        })

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
            Pi_B_out = self._construct_Pi_B_out(Pi_out, Lambda_l_in)
            Pi_A_out = Pi_out - Pi_B_out
            Pi_B_in = self._construct_Pi_B_in(Pi_in, Lambda_l_out, Pi_B_out)
            Pi_A_in = Pi_in - Pi_B_in

            Pi_bar_out = self._smallest_superset(l, self.basis_out, Pi_B_out,
                                                 Lambda_l_out)
            Pi_bar_in = self._smallest_superset(l, self.basis_in, Pi_B_in,
                                                Lambda_l_in)
            d_bar = self.basis_in.P.matvec(Pi_B_in, Pi_bar_in, d) + \
                    self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            e_bar, f_bar = self._apply_recur(l + 1, Pi_bar_in, Pi_bar_out,
                                             d_bar, c)
            e = self.operator.matvec(Pi_in, Pi_A_out, d) + \
                self.basis_out.P.rmatvec(Pi_bar_out, Pi_B_out, e_bar)
            f = self.basis_out.Q.rmatvec(Pi_bar_out, Lambda_l_out,
                                         e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_upp_recur(self, l, Pi_in, Pi_out, d, c):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Pi_out) + len(Lambda_l_out) > 0 and len(Pi_in) + len(
                Lambda_l_in) > 0:
            Pi_B_out = self._construct_Pi_B_out(Pi_out, Lambda_l_in)
            Pi_A_out = Pi_out - Pi_B_out

            Pi_bar_out = self._smallest_superset(l, self.basis_out, Pi_B_out,
                                                 Lambda_l_out)
            Pi_bar_in = self._smallest_superset(l,
                                                self.basis_in,
                                                Pi_B={},
                                                Lambda_l=Lambda_l_in)

            d_bar = self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            e_bar, f_bar = self._apply_upp_recur(l + 1, Pi_bar_in, Pi_bar_out,
                                                 d_bar, c)
            e = self.operator.matvec(Pi_in, Pi_out, d) + \
                self.basis_out.P.rmatvec(Pi_bar_out, Pi_B_out, e_bar)
            f = self.basis_out.Q.rmatvec(Pi_bar_out, Lambda_l_out,
                                         e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_low_recur(self, l, Pi_in, d, c):
        Lambda_l_in = self.Lambda_in.on_level(l)
        Lambda_l_out = self.Lambda_out.on_level(l)
        if len(Lambda_l_out) > 0 and len(Pi_in) + len(Lambda_l_in) > 0:
            Pi_B_in = self._construct_Pi_B_in(Pi_in, Lambda_l_out, Pi_B_out={})
            Pi_bar_in = self._smallest_superset(l, self.basis_in, Pi_B_in,
                                                Lambda_l_in)
            Pi_B_bar_in = self._smallest_superset(l,
                                                  self.basis_in,
                                                  Pi_B=Pi_B_in,
                                                  Lambda_l={})
            Pi_B_bar_out = self._smallest_superset(l,
                                                   self.basis_out,
                                                   Pi_B={},
                                                   Lambda_l=Lambda_l_out)

            # NB: operator is applied at level `l` -- different from the rest.
            e_bar = self.operator.matvec(
                Pi_B_bar_in, Pi_B_bar_out,
                self.basis_in.P.matvec(Pi_B_in, Pi_B_bar_in, d))
            d_bar = self.basis_in.P.matvec(Pi_B_in, Pi_bar_in, d) + \
                    self.basis_in.Q.matvec(Lambda_l_in, Pi_bar_in, c)
            f = self.basis_out.Q.rmatvec(Pi_B_bar_out, Lambda_l_out,
                                         e_bar) + self._apply_low_recur(
                                             l + 1, Pi_bar_in, d_bar, c)
            return f
        else:
            return IndexedVector.Zero()
