from index_set import IndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet


class Applicator(object):
    """ Class that can apply multiscale operators with minimal overhead. """

    def __init__(self, basis, A_ss, Lambda):
        """ Initialize the applicator.

        Arguments:
            basis: A Basis object.
            A_ss: the singlescale operator (eg basis.singlescale_mass).
            Lambda: the multiscale indices to apply the multiscale operator on.
        """
        self.basis = basis
        self.operator = A_ss
        self.Lambda = Lambda

    def apply(self, vec):
        """ Apply the multiscale operator.

        Arguments:
            vec: an IndexedVector with indices on self.Lambda.

        Returns: self.operator(Psi_Lambda)(Psi_Lambda) vec.
        """
        e, f = self._apply_recur(l=1,
                                 Pi=self.Lambda.on_level(0),
                                 d=vec.on_level(0),
                                 c=vec.from_level(1))
        return e + f

    def apply_upp(self, vec):
        e, f = self._apply_upp_recur(l=1,
                                     Pi=self.Lambda.on_level(0),
                                     d=vec.on_level(0),
                                     c=vec.from_level(1))
        return e + f

    def apply_low(self, vec):
        f = self._apply_low_recur(l=1,
                                  Pi=self.Lambda.on_level(0),
                                  d=vec.on_level(0),
                                  c=vec.from_level(1))
        return f

    #  Private methods from here on out.

    def _apply_P(self, Pi_B, Pi_bar, d):
        """ Apply P_l: ell_2(Pi_B) to ell_2(Pi_bar).

        P_l is the matrix for which Phi_{l-1}^top = Phi_l^top P_l.
        It is the matrix corresponding with embedding sp Phi_{l-1} in sp Phi_l.

        Arguments:
            Pi_B: the single-scale indices on the previous level
            Pi_bar: the single-scale indices on this level
            d: a vector with nonzero coefficients only on Pi_B.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        res = IndexedVector({
            labda: sum([
                d[k] * v for (k, v) in zip(self.basis.scaling_parents(labda),
                                           self.basis.P_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def _apply_Q(self, Lambda_l, Pi_bar, c):
        """ Apply Q: ell_2(Lambda_l) to ell_2(Pi_bar).

        This is the matrix corresponding with embedding sp Psi_l in sp Phi_l.

        Arguments:
            Lambda_l: the multiscale indices on this level
            Pi_bar: the single-scale indices on this level
            c: a vector with nonzero coefficients only on Lambda_l.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        res = IndexedVector({
            labda: sum([
                c[k] * v for (k, v) in zip(self.basis.scaling_siblings(labda),
                                           self.basis.Q_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def _apply_PT(self, Pi_bar, Pi_B, e_bar):
        """ Apply P^top: ell_2(Pi_bar) to ell_2(Pi_B). """
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v
                for (k, v) in zip(self.basis.scaling_children(labda),
                                  self.basis.PT_block(labda))
            ])
            for labda in Pi_B
        })
        return res

    def _apply_QT(self, Pi_bar, Lambda_l, e_bar):
        """ Apply Q^top: ell_2(Pi_bar) to ell_2(Lambda_l). """
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v
                for (k, v) in zip(self.basis.wavelet_siblings(labda),
                                  self.basis.QT_block(labda))
            ])
            for labda in Lambda_l
        })
        return res

    def _singlescale_supports_intersecting(self, l, Pi, Lambda_l):
        return IndexSet({
            index
            for index in Pi if any([
                self.basis.scaling_support(index).intersects(
                    self.basis.wavelet_nbrhood(mu)) for mu in Lambda_l
            ])
        })

    def _singlescale_supports_covered_by(self, l, interval_set):
        return IndexSet({
            index
            for index in self.basis.scaling_indices_on_level(l)
            if interval_set.covers(self.basis.scaling_support(index))
        })

    def _apply_recur(self, l, Pi, d, c):
        """ Apply the multiscale operator on level l.
        
        Arguments:
            Pi: a collection of singlescale indices on level l-1.
            d: a vector in ell_2(Pi).
            c: a vector in ell_2(Lambda_{l up}).

        Output:
            e: a vector in ell_2(Pi_bar)
            f: a vector in ell_2(Lambda_{l+1 up}).
        """
        Lambda_lup = self.Lambda.from_level(l)
        if len(Pi) + len(Lambda_lup) > 0:
            Lambda_l = Lambda_lup.on_level(l)
            Pi_B = self._singlescale_supports_intersecting(l, Pi, Lambda_l)
            Pi_A = Pi.difference(Pi_B)
            ivs_all = IntervalSet(
                [self.basis.scaling_support(index) for index in Pi_B] +
                [self.basis.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self._singlescale_supports_covered_by(l, ivs_all)
            d_bar = self._apply_P(Pi_B, Pi_bar, d) + \
                    self._apply_Q(Lambda_l, Pi_bar, c)
            e_bar, f_bar = self._apply_recur(l + 1, Pi_bar, d_bar,
                                             c.from_level(l + 1))
            e = self.operator(l - 1, Pi, Pi_A, d) + \
                self._apply_PT(Pi_bar, Pi_B, e_bar)
            f = self._apply_QT(Pi_bar, Lambda_l, e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_upp_recur(self, l, Pi, d, c):
        Lambda_lup = self.Lambda.from_level(l)
        Lambda_l = Lambda_lup.on_level(l)
        if len(Pi) + len(Lambda_l) > 0:
            Pi_B = self._singlescale_supports_intersecting(l, Pi, Lambda_l)
            Pi_A = Pi.difference(Pi_B)
            ivs_multi = IntervalSet(
                [self.basis.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self._singlescale_supports_covered_by(l, ivs_multi)
            d_bar = self._apply_Q(Lambda_l, Pi_bar, c)
            e_bar, f_bar = self._apply_upp_recur(l + 1, Pi_bar, d_bar,
                                                 c.from_level(l + 1))
            e = self.operator(l - 1, Pi, Pi, d) + \
                self._apply_PT(Pi_bar, Pi_B, e_bar)
            f = self._apply_QT(Pi_bar, Lambda_l, e_bar) + f_bar
            return e, f
        else:
            return IndexedVector.Zero(), IndexedVector.Zero()

    def _apply_low_recur(self, l, Pi, d, c):
        Lambda_lup = self.Lambda.from_level(l)
        Lambda_l = Lambda_lup.on_level(l)
        if len(Pi) + len(Lambda_l) > 0:
            Pi_B = self._singlescale_supports_intersecting(l, Pi, Lambda_l)
            ivs_all = IntervalSet(
                [self.basis.scaling_support(index) for index in Pi_B] +
                [self.basis.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self._singlescale_supports_covered_by(l, ivs_all)
            ivs_single = IntervalSet(
                [self.basis.wavelet_support(index) for index in Lambda_l])
            Pi_B_bar = self._singlescale_supports_covered_by(l, ivs_single)
            ivs_multi = IntervalSet(
                [self.basis.scaling_support(index) for index in Pi_B])
            Pi_B_bar_flup = self._singlescale_supports_covered_by(l, ivs_multi)

            # NB: operator is applied at level `l` -- difference w/ apply_recur.
            e_bar = self.operator(l, Pi_B_bar, Pi_B_bar_flup,
                                  self._apply_P(Pi_B, Pi_B_bar, d))
            d_bar = self._apply_P(Pi_B, Pi_bar, d) + \
                    self._apply_Q(Lambda_l, Pi_bar, c)
            f = self._apply_QT(Pi_B_bar, Lambda_l, e_bar) + \
                self._apply_low_recur(l + 1, Pi_bar, d_bar, c.from_level(l + 1))
            return f
        else:
            return IndexedVector.Zero()
