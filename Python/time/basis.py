from index_set import IndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
import numpy as np

sq3 = np.sqrt(3)


class Basis(object):
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
            labda: 2**-l * d[labda] if labda in Pi_A else 0.0
            for labda in Pi
        })
        return res

    def singlescale_damping(self, l, Pi, Pi_A, d):
        """ The singlescale damping matrix int_0^1 phi_i phi_j' dt. """
        assert len(Pi) == 0 or next(iter(Pi))[0] == l  #Pi is on level l only.
        res = IndexedVector({
            labda:
            2 * sq3 * d[(labda[0], labda[1] + 1)] if labda[1] % 2 == 0 else 0.0
            for labda in Pi
        })
        return res
