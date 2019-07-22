from index_set import IndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
import numpy as np

sq3 = np.sqrt(3)


class Basis(object):
    def eval_scaling(self, labda, x):
        """ Debug method. """
        pass

    def eval_wavelet(self, labda, x):
        """ Debug method. """
        pass

    def scaling_parents(self, index):
        """ Parent singlescale indices overlapping this singlescale index. """
        pass

    def scaling_siblings(self, index):
        """ Multiscale indices interacting with this singlescale index. """
        pass

    def scaling_children(self, index):
        """ Children singlescale indices overlapping this singlescale index. """
        pass

    def wavelet_siblings(self, index):
        """ Singlescale indices interacting with this multiscale index. """
        pass

    def P_block(self, labda):
        """ The vector st phi_{labda} = Phi_ell^top P_block (|labda| = ell). """
        pass

    def Q_block(self, labda):
        """ The vector st psi_{labda} = Phi_ell^top Q_block (|labda| = ell). """
        pass

    def PT_block(self, labda):
        """ The vector st phi_{labda} = P_block^top Phi_ell (|labda| = ell). """
        pass

    def QT_block(self, labda):
        """ The vector st psi_{labda} = Q_block^top Phi_ell (|labda| = ell). """
        pass

    def apply_P(self, Pi_B, Pi_bar, d):
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
                d[k] * v if k in Pi_B else 0.0 for (
                    k,
                    v) in zip(self.scaling_parents(labda), self.P_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def apply_Q(self, Lambda_l, Pi_bar, c):
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
                c[k] * v if k in Lambda_l else 0.0 for (k, v) in zip(
                    self.scaling_siblings(labda), self.Q_block(labda))
            ])
            for labda in Pi_bar
        })
        return res

    def apply_PT(self, Pi_bar, Pi_B, e_bar):
        """ Apply P^top: ell_2(Pi_bar) to ell_2(Pi_B). """
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                    self.scaling_children(labda), self.PT_block(labda))
            ])
            for labda in Pi_B
        })
        return res

    def apply_QT(self, Pi_bar, Lambda_l, e_bar):
        """ Apply Q^top: ell_2(Pi_bar) to ell_2(Lambda_l). """
        res = IndexedVector({
            labda: sum([
                e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                    self.wavelet_siblings(labda), self.QT_block(labda))
            ])
            for labda in Lambda_l
        })
        return res

    def scaling_support(self, labda):
        """ The support of the scaling function phi_labda. """
        pass

    def wavelet_support(self, labda):
        """ The support of the wavelet function psi_labda. """
        pass

    def wavelet_nbrhood(self, labda):
        """ The neighbourhood S(psi_labda). Defaults to its support. """
        return self.wavelet_support(labda)

    def scaling_indices_on_level(self, l):
        """ IndexSet of singlescale indices on level l. """
        pass

    def uniform_wavelet_indices(self, max_level):
        """ IndexSet of multiscale indices wrt uniform refinement. """
        pass

    def origin_refined_wavelet_indices(self, max_level):
        """ IndexSet of multiscale indices wrt refinement towards origin. """
        pass


class HaarBasis(Basis):
    def eval_mother_scaling(self, x):
        return (0 <= x) & (x < 1)

    def eval_scaling(self, labda, x):
        return 1.0 * self.eval_mother_scaling(2**(labda[0]) * x - labda[1])

    def eval_mother_wavelet(self, x):
        return 1.0 * ((0 <= x) & (x < 0.5)) - 1.0 * ((0.5 <= x) & (x < 1.0))

    def eval_wavelet(self, labda, x):
        if labda[0] == 0:
            return 1.0 * self.eval_mother_scaling(x)
        else:
            return 1.0 * self.eval_mother_wavelet(2**(labda[0] - 1) * x -
                                                  labda[1])

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
            labda: 2**-l * d[labda] if labda in Pi else 0.0
            for labda in Pi_A
        })
        return res


class OrthonormalDiscontinuousLinearBasis(Basis):
    """ We have a multiwavelet basis.

    It has two wavelets and scaling functions. Even index-offsets correspond
    with the first, odd with the second.
    """

    def eval_mother_scaling(self, odd, x):
        if odd:
            return sq3 * (2 * x - 1) * ((0 <= x) & (x < 1))
        else:
            return (0 <= x) & (x < 1)

    def eval_scaling(self, labda, x):
        return 1.0 * self.eval_mother_scaling(
            labda[1] % 2, 2**(labda[0]) * x - (labda[1] // 2))

    def eval_mother_wavelet(self, odd, x):
        if odd:
            return sq3 * (1 - 4 * x) * (
                (0 <= x) & (x < 0.5)) + sq3 * (4 * x - 3) * ((0.5 <= x) &
                                                             (x < 1))
        else:
            return (1 - 6 * x) * ((0 <= x) &
                                  (x < 0.5)) + (5 - 6 * x) * ((0.5 <= x) &
                                                              (x < 1))

    def eval_wavelet(self, labda, x):
        if labda[0] == 0:
            return 1.0 * self.eval_mother_scaling(labda[1] % 1, x)
        else:
            return 2**((labda[0] - 1) / 2) * self.eval_mother_wavelet(
                labda[1] % 2, 2**(labda[0] - 1) * x - (labda[1] // 2))

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
        assert index[0] > 0
        return [(index[0] - 1, 2 * (index[1] // 4) + i) for i in range(2)]

    def scaling_siblings(self, index):
        return [(index[0], 2 * (index[1] // 4) + i) for i in range(2)]

    def scaling_children(self, index):
        return [(index[0] + 1, 4 * (index[1] // 2) + i) for i in range(4)]

    def wavelet_siblings(self, index):
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
            labda: 2**-l * d[labda] if labda in Pi else 0.0
            for labda in Pi_A
        })
        return res

    def singlescale_damping(self, l, Pi, Pi_A, d):
        """ The singlescale damping matrix int_0^1 phi_i phi_j' dt. """
        assert len(Pi) == 0 or next(iter(Pi))[0] == l  #Pi is on level l only.
        res = IndexedVector({
            labda: 2 * sq3 * d[(labda[0], labda[1] + 1)]
            if labda[1] % 2 == 0 and labda in Pi else 0.0
            for labda in Pi_A
        })
        return res
