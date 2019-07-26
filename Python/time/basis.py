from index_set import IndexSet
from indexed_vector import IndexedVector
from interval import Interval, IntervalSet
import numpy as np

sq3 = np.sqrt(3)


class Basis(object):
    def __init__(self, indices):
        """ Default constructor. """
        self.indices = indices

    @classmethod
    def uniform_basis(cls, max_level):
        """ Constructor for basis wrt uniform refinement. """
        return cls(indices=cls._uniform_multilevel_indices(max_level))

    @classmethod
    def _uniform_multilevel_indices(cls, max_level):
        """ IndexSet of multiscale indices wrt uniform refinement. """
        pass

    @classmethod
    def origin_refined_basis(cls, max_level):
        """ Constructor for basis wrt refinement towards origin. """
        return cls(indices=cls._origin_refined_multilevel_indices(max_level))

    @classmethod
    def _origin_refined_multilevel_indices(cls, max_level):
        """ IndexSet of multiscale indices wrt refinement towards origin. """
        pass

    def eval_scaling(self, labda, x):
        """ Debug method. """
        pass

    def eval_wavelet(self, labda, x):
        """ Debug method. """
        pass

    def P_block(self, labda):
        """ The vector st phi_{labda} = Phi_ell^T P_block (|labda| = ell). """
        pass

    def scaling_parents(self, index):
        """ Parent singlescale indices that build this singlescale index.
        
        Basically the indices of the nonzero elements of P_l[:,index].
        """
        pass

    def Q_block(self, labda):
        """ The vector st psi_{labda} = Phi_ell^T Q_block (|labda| = ell). """
        pass

    def scaling_siblings(self, index):
        """ MS-indices on this level that interact with this SS-index.
        
        Basically the indices of the nonzero elements of Q_l[:,index].
        """
        pass

    def PT_block(self, labda):
        """ The vector st phi_{labda} = P_block^T Phi_ell (|labda| = ell). """
        pass

    def scaling_children(self, index):
        """ Children singlescale indices overlapping this singlescale index.
        
        Basically the indices of the nonzero elements of P_l^T[:,index].
        """
        pass

    def QT_block(self, labda):
        """ The vector st psi_{labda} = Q_block^T Phi_ell (|labda| = ell). """
        pass

    def wavelet_siblings(self, index):
        """ Singlescale indices interacting with this multiscale index.
        
        Basically the indices of the nonzero elements of Q_l^T[:,index].
        """
        pass

    def apply_P(self, Pi_B, Pi_bar, d, out=None):
        """ Apply P_l: ell_2(Pi_B) to ell_2(Pi_bar).

        P_l is the matrix for which Phi_{l-1}^T = Phi_l^T P_l.
        It is the matrix corresponding with embedding sp Phi_{l-1} in sp Phi_l.

        Arguments:
            Pi_B: the single-scale indices on the previous level
            Pi_bar: the single-scale indices on this level
            d: IndexedVector on (a superset of) Pi_B.
            out: (optional) the output vector.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        if not out:
            res = IndexedVector({
                labda: sum([
                    d[k] * v if k in Pi_B else 0.0 for (k, v) in zip(
                        self.scaling_parents(labda), self.P_block(labda))
                ])
                for labda in Pi_bar
            })
            return res
        else:
            for labda in Pi_bar:
                out[labda] = sum([
                    d[k] * v if k in Pi_B else 0.0 for (k, v) in zip(
                        self.scaling_parents(labda), self.P_block(labda))
                ])

    def apply_Q(self, Lambda_l, Pi_bar, c, out=None):
        """ Apply Q: ell_2(Lambda_l) to ell_2(Pi_bar).

        Q_l is the matrix for which Psi_l^T = Phi_l^T Q_l.
        This is the matrix corresponding with embedding sp Psi_l in sp Phi_l.

        Arguments:
            Lambda_l: the multiscale indices on this level
            Pi_bar: the single-scale indices on this level
            c: a vector with nonzero coefficients only on Lambda_l.
        Output:
            res: a vector with nonzero coefficients only on Pi_bar.
        """
        if not out:
            res = IndexedVector({
                labda: sum([
                    c[k] * v if k in Lambda_l else 0.0 for (k, v) in zip(
                        self.scaling_siblings(labda), self.Q_block(labda))
                ])
                for labda in Pi_bar
            })
            return res
        else:
            for labda in Pi_bar:
                labda: sum([
                    c[k] * v if k in Lambda_l else 0.0 for (k, v) in zip(
                        self.scaling_siblings(labda), self.Q_block(labda))
                ])

    def apply_PT(self, Pi_bar, Pi_B, e_bar, out=None):
        """ Apply P^T: ell_2(Pi_bar) to ell_2(Pi_B). """
        if not out:
            res = IndexedVector({
                labda: sum([
                    e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                        self.scaling_children(labda), self.PT_block(labda))
                ])
                for labda in Pi_B
            })
            return res
        else:
            for labda in Pi_B:
                out[labda] = sum([
                    e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                        self.scaling_children(labda), self.PT_block(labda))
                ])

    def apply_QT(self, Pi_bar, Lambda_l, e_bar, out=None):
        """ Apply Q^T: ell_2(Pi_bar) to ell_2(Lambda_l). """
        if not out:
            res = IndexedVector({
                labda: sum([
                    e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                        self.wavelet_siblings(labda), self.QT_block(labda))
                ])
                for labda in Lambda_l
            })
            return res
        else:
            for labda in Lambda_l:
                out[labda] = sum([
                    e_bar[k] * v if k in Pi_bar else 0.0 for (k, v) in zip(
                        self.wavelet_siblings(labda), self.QT_block(labda))
                ])

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
        """ SingleLevelIndexSet of singlescale indices on level l. """
        pass
