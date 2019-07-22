from index import IndexSet, IndexedVector
from interval import Interval, IntervalSet
import numpy as np

sq3 = np.sqrt(3)


class Basis(object):
    def scaling_parents(self, index):
        pass

    def scaling_siblings(self, index):
        pass

    def scaling_children(self, index):
        pass

    def wavelet_siblings(self, index):
        pass

    def P_block(self, labda):
        pass

    def apply_P(self, l, Pi_B, Pi_bar, d):
        """ Apply P_l: ell_2(Pi_B) to ell_2(Pi_bar).

        P_l is the matrix for which Phi_{l-1}^top = Phi_l^top P_l.
        It is the matrix corresponding with embedding sp Phi_{l-1} in sp Phi_l.

        Arguments:
            l: the current level
            Pi_B: the single-scale indices on the previous level
            Pi_bar: the single-scale indices on this level
            d: a vector with nonzero coefficients only on Pi_B.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        assert l > 0
        res = IndexedVector({
            labda: sum([
                d[k] * v for (k, v) in zip(self.scaling_parents(labda), \
                                           self.P_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def Q_block(self, labda):
        pass

    def apply_Q(self, l, Lambda_l, Pi_bar, c):
        """ Apply Q: ell_2(Lambda_l) to ell_2(Pi_bar).

        This is the matrix corresponding with embedding sp Psi_l in sp Phi_l.

        Arguments:
            l: the current level
            Lambda_l: the multiscale indices on this level
            Pi_bar: the single-scale indices on this level
            c: a vector with nonzero coefficients only on Lambda_l.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        assert l > 0
        res = IndexedVector({
            labda: sum([
                c[k] * v for (k, v) in zip(self.scaling_siblings(labda), \
                                           self.Q_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def PT_block(self, labda):
        pass

    def apply_PT(self, l, Pi_bar, Pi_B, e_bar):
        """ Apply P^top: ell_2(Pi_bar) to ell_2(Pi_B). """
        assert l > 0
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v for (k, v) in zip(self.scaling_children(labda), \
                                               self.PT_block(labda))
            ])
            for labda in Pi_B
        })
        return res

    def QT_block(self, labda):
        pass

    def apply_QT(self, l, Pi_bar, Lambda_l, e_bar):
        """ Apply Q^top: ell_2(Pi_bar) to ell_2(Lambda_l). """
        assert l > 0
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v for (k, v) in zip(self.wavelet_siblings(labda), \
                                               self.QT_block(labda))
            ])
            for labda in Lambda_l
        })
        return res

    def scaling_support(self, index):
        pass

    def wavelet_support(self, index):
        pass

    def wavelet_nbrhood(self, index):
        pass

    def scaling_indices_on_level(self, l):
        pass

    def uniform_wavelet_indices(self, max_level):
        pass

    def origin_refined_wavelet_indices(self, max_level):
        pass

    def singlescale_mass(self, l, Pi, Pi_A, d):
        """ The singlescale mass matrix int_0^1 phi_i phi_j dt. """
        pass

    def singlescale_damping(self, l, Pi, Pi_A, d):
        """ The singlescale damping matrix int_0^1 phi_i' phi_j dt. """
        pass

    def singlescale_supports_intersecting(self, l, Pi, Lambda_l):
        return IndexSet({
            index
            for index in Pi if any([
                self.scaling_support(index).intersects(self.wavelet_nbrhood(
                    mu)) for mu in Lambda_l
            ])
        })

    def singlescale_supports_covered_by(self, l, interval_set):
        return IndexSet({
            index
            for index in self.scaling_indices_on_level(l)
            if interval_set.covers(self.scaling_support(index))
        })

    def apply_operator_recur(self, operator, l, Pi, Lambda_lup, d, c):
        """ Bleep bloop TODO.
        
        Arguments:
            Pi: a collection of singlescale indices on level l.
            Lambda_lup: a collection of multiscale indices, an l-tree.
            d: a vector in ell_2(Pi).
            c: a vector in ell_2(Lambda_lup).

        Output:
            e: a vector in ell_2(Pi_bar)
            f: a vector in ell_2(Lambda_{l+1 up}).
        """
        if len(Pi) + len(Lambda_lup) > 0:
            Lambda_l = Lambda_lup.on_level(l)
            Pi_B = self.singlescale_supports_intersecting(l, Pi, Lambda_l)
            Pi_A = Pi.difference(Pi_B)
            ivs_all = IntervalSet(
                [self.scaling_support(index) for index in Pi_B] +
                [self.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self.singlescale_supports_covered_by(l, ivs_all)
            d_bar = IndexedVector.sum(self.apply_P(l, Pi_B, Pi_bar, d),
                                      self.apply_Q(l, Lambda_l, Pi_bar, c))
            e_bar, f_bar = self.apply_operator_recur(
                operator, l + 1, Pi_bar, Lambda_lup.from_level(l + 1), d_bar,
                c.from_level(l + 1))
            e = IndexedVector.sum(operator(l - 1, Pi, Pi_A, d),
                                  self.apply_PT(l, Pi_bar, Pi_B, e_bar))
            f = IndexedVector.sum(self.apply_QT(l, Pi_bar, Lambda_l, e_bar),
                                  f_bar)
            return e, f
        else:
            return d, c

    def apply_operator_upp_recur(self, operator, l, Pi, Lambda_lup, d, c):
        """ Bleep bloop TODO. """
        Lambda_l = Lambda_lup.on_level(l)
        if len(Pi) + len(Lambda_l) > 0:
            Pi_B = self.singlescale_supports_intersecting(l, Pi, Lambda_l)
            Pi_A = Pi.difference(Pi_B)
            ivs_multi = IntervalSet(
                [self.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self.singlescale_supports_covered_by(l, ivs_multi)
            d_bar = self.apply_Q(l, Lambda_l, Pi_bar, c)
            e_bar, f_bar = self.apply_operator_upp_recur(
                operator, l + 1, Pi_bar, Lambda_lup.from_level(l + 1), d_bar,
                c.from_level(l + 1))
            e = IndexedVector.sum(operator(l - 1, Pi, Pi, d),
                                  self.apply_PT(l, Pi_bar, Pi_B, e_bar))
            f = IndexedVector.sum(self.apply_QT(l, Pi_bar, Lambda_l, e_bar),
                                  f_bar)
            return e, f
        else:
            return d, c

    def apply_operator_low_recur(self, operator, l, Pi, Lambda_lup, d, c):
        """ Bleep bloop TODO. """
        Lambda_l = Lambda_lup.on_level(l)
        if len(Pi) + len(Lambda_l) > 0:
            Pi_B = self.singlescale_supports_intersecting(l, Pi, Lambda_l)
            ivs_all = IntervalSet(
                [self.scaling_support(index) for index in Pi_B] +
                [self.wavelet_support(index) for index in Lambda_l])
            Pi_bar = self.singlescale_supports_covered_by(l, ivs_all)
            ivs_single = IntervalSet(
                [self.wavelet_support(index) for index in Lambda_l])
            Pi_B_bar = self.singlescale_supports_covered_by(l, ivs_single)
            ivs_multi = IntervalSet(
                [self.scaling_support(index) for index in Pi_B])
            Pi_B_bar_flup = self.singlescale_supports_covered_by(l, ivs_multi)
            # NB: operator is applied at level `l` -- difference with evalA.
            e_bar = operator(l, Pi_B_bar, Pi_B_bar_flup,
                             self.apply_P(l, Pi_B, Pi_B_bar, d))
            d_bar = IndexedVector.sum(self.apply_P(l, Pi_B, Pi_bar, d),
                                      self.apply_Q(l, Lambda_l, Pi_bar, c))
            f = IndexedVector.sum(
                self.apply_QT(l, Pi_B_bar, Lambda_l, e_bar),
                self.apply_operator_low_recur(operator, l + 1, Pi_bar,
                                              Lambda_lup.from_level(l + 1),
                                              d_bar, c.from_level(l + 1)))
            return f
        else:
            return c

    def apply_operator(self, operator, Lambda, c):
        Lambda0 = Lambda.on_level(0)
        Lambda1up = Lambda.from_level(1)
        e, f = self.apply_operator_recur(operator, 1, Lambda0, Lambda1up,
                                         c.on_level(0), c.from_level(1))
        return IndexedVector.sum(e, f)

    def apply_operator_upp(self, operator, Lambda, c):
        Lambda0 = Lambda.on_level(0)
        Lambda1up = Lambda.from_level(1)
        e, f = self.apply_operator_upp_recur(operator, 1, Lambda0, Lambda1up,
                                             c.on_level(0), c.from_level(1))
        return IndexedVector.sum(e, f)

    def apply_operator_low(self, operator, Lambda, c):
        Lambda0 = Lambda.on_level(0)
        Lambda1up = Lambda.from_level(1)
        f = self.apply_operator_low_recur(operator, 1, Lambda0, Lambda1up,
                                          c.on_level(0), c.from_level(1))
        return f


class HaarBasis(Basis):
    def wavelet_support(self, index):
        if index[0] == 0:
            assert index[1] == 0
            return Interval(0, 1)
        else:
            assert 0 <= index[1] < 2**(index[0] - 1)
            return Interval(2**-(index[0] - 1) * index[1],
                            2**-(index[0] - 1) * (index[1] + 1))

    def wavelet_nbrhood(self, index):
        return self.wavelet_support(index)

    def wavelet_indices_on_level(self, l):
        if l == 0:
            return IndexSet({0, 0})
        else:
            return IndexSet({(l, n) for n in range(2**(l - 1))})

    def scaling_support(self, index):
        return Interval(2**-index[0] * index[1], 2**-index[0] * (index[1] + 1))

    def scaling_parents(self, index):
        return [(index[0] - 1, index[1] // 2)]

    def scaling_siblings(self, index):
        return [(index[0], index[1] // 2)]

    def scaling_children(self, index):
        return [(index[0] + 1, 2 * index[1]), (index[0] + 1, 2 * index[1] + 1)]

    def wavelet_siblings(self, index):
        return [(index[0], 2 * index[1]), (index[0], 2 * index[1] + 1)]

    def scaling_indices_on_level(self, l):
        return IndexSet({(l, n) for n in range(2**l)})

    def uniform_wavelet_indices(self, max_level):
        return IndexSet({(0, 0)} | {(l, n)
                                    for l in range(1, max_level + 1)
                                    for n in range(2**(l - 1))})

    def origin_refined_wavelet_indices(self, max_level):
        return IndexSet({(l, 0) for l in range(max_level + 1)})

    def P_block(self, labda):
        return [1]

    def Q_block(self, labda):
        return [(-1)**labda[1]]

    def PT_block(self, labda):
        return [1, 1]

    def QT_block(self, labda):
        return [1, -1]

    def singlescale_mass(self, l, Pi, Pi_A, d):
        """ The singlescale Haar mass matrix is simply 2**-l * Id. """
        assert len(Pi) == 0 or next(iter(Pi))[0] == l  #Pi is on level l only.
        res = IndexedVector({
            labda: 2**-l * d[labda] if labda in Pi_A else 0.0
            for labda in Pi
        })
        return res


class OrthonormalDiscontinuousLinearBasis(Basis):
    """ We have a multiwavelet basis.

    It has two wavelets and scaling functions. Even index-offsets correspond
    with the first, odd with the second.
    """

    def wavelet_support(self, index):
        if index[0] == 0:
            assert index[1] in [0, 1]
            return Interval(0, 1)
        else:
            assert 0 <= index[1] < 2 * 2**(index[0] - 1)
            return Interval(2**-(index[0] - 1) * index[1] // 2,
                            2**-(index[0] - 1) * (index[1] // 2 + 1))

    def wavelet_nbrhood(self, index):
        return self.wavelet_support(index)

    def wavelet_indices_on_level(self, l):
        if l == 0:
            return IndexSet({(0, 0), (0, 1)})
        else:
            return IndexSet({(l, n) for n in range(2 * 2**(l - 1))})

    def scaling_support(self, index):
        return Interval(2**-index[0] * index[1] // 2,
                        2**-index[0] * (index[1] // 2 + 1))

    def scaling_parents(self, index):
        """ Parent singlescale indices overlapping this singlescale index. """
        assert index[0] > 0
        return [(index[0] - 1, 2 * (index[1] // 4) + i) for i in range(2)]

    def scaling_siblings(self, index):
        """ Multiscale indices interacting with this singlescale index. """
        return [(index[0], 2 * (index[1] // 4) + i) for i in range(2)]

    def scaling_children(self, index):
        """ Children singlescale indices overlapping this singlescale index. """
        return [(index[0] + 1, 4 * (index[1] // 2) + i) for i in range(4)]

    def wavelet_siblings(self, index):
        """ Singlescale indices interacting with this multiscale index. """
        return [(index[0], 4 * (index[1] // 2) + i) for i in range(4)]

    def scaling_indices_on_level(self, l):
        return IndexSet({(l, n) for n in range(2 * 2**l)})

    def uniform_wavelet_indices(self, max_level):
        return IndexSet({(0, 0), (0, 1)} | {(l, n)
                                            for l in range(1, max_level + 1)
                                            for n in range(2 * 2**(l - 1))})

    def origin_refined_wavelet_indices(self, max_level):
        return IndexSet({(l, i)
                         for l in range(max_level + 1) for i in range(2)})

    P_mask = np.array([[1, 0, 1, 0], [-sq3 / 2, 1 / 2.0, sq3 / 2, 1 / 2.0]])

    Q_mask = np.array([[-1 / 2.0, -sq3 / 2, 1 / 2.0, -sq3 / 2], [0, -1, 0, 1]])

    def P_block(self, labda):
        return self.P_mask[:, labda[1] % 4]

    def Q_block(self, labda):
        return 2.0**((labda[0] - 1) / 2) * self.Q_mask[:, labda[1] % 4]

    def PT_block(self, labda):
        return self.P_mask[labda[1] % 2, :]

    def QT_block(self, labda):
        return 2.0**((labda[0] - 1) / 2) * self.Q_mask[labda[1] % 2, :]

    def singlescale_mass(self, l, Pi, Pi_A, d):
        assert len(Pi) == 0 or next(iter(Pi))[0] == l  #Pi is on level l only.
        res = IndexedVector({
            labda: 2**-l * d[labda] if labda in Pi_A else 0.0
            for labda in Pi
        })
        return res

    def singlescale_damping(self, l, Pi, Pi_A, d):
        """ The singlescale damping matrix int_0^1 phi_i' phi_j dt. """
        assert len(Pi) == 0 or next(iter(Pi))[0] == l  #Pi is on level l only.
        res = IndexedVector({
            labda:
            2 * sq3 * d[(labda[0], labda[1] + 1)] if labda[1] % 2 == 0 else 0.0
            for labda in Pi
        })
        return res
