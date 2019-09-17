import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator


class Operators:
    """ (Temporary) class that defines operators on a triangulation. """
    def __init__(self, triang):
        self.triang = triang

    def apply_T(self, v):
        """
        Applies the hierarchical-to-single-scale transformation to a vector `v`.

        Arguments:
            v: a `np.array` of length len(self.vertices).

        Returns:
            w: a `np.array` of length len(self.vertices).
        """

        w = np.copy(v)
        for (vi, Ti) in self.triang.history:
            godparent_vertices = [
                v.idx for v in self.triang.elements[Ti].edge(0)
            ]
            for gp in godparent_vertices:
                w[vi] = w[vi] + 0.5 * w[gp]
        return w

    def apply_T_transpose(self, v):
        """
        Applies the transposed hierarchical-to-single-scale transformation to `v`.

        Arguments:
            v: a `np.array` of length len(self.triang.vertices).

        Returns:
            w: a `np.array` of length len(self.triang.vertices).
        """
        w = np.copy(v)
        for (vi, Ti) in reversed(self.triang.history):
            godparent_vertices = [
                v.idx for v in self.triang.elements[Ti].edge(0)
            ]
            for gp in godparent_vertices:
                w[gp] = w[gp] + 0.5 * w[vi]
        return w

    def apply_SS_mass(self, v):
        """ Applies the single-scale mass matrix. """
        element_mass = 1.0 / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue

            Vids = [v.idx for v in elem.vertices]
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_mass[i, j] * elem.area * v[Vids[i]]
        return w

    def apply_SS_stiffness(self, v):
        """ Applies the single-scale stiffness matrix. """
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue
            V = elem.vertices
            Vids = [v.idx for v in V]
            D = np.array([[V[2].x - V[1].x, V[0].x - V[2].x, V[1].x - V[0].x],
                          [V[2].y - V[1].y, V[0].y - V[2].y, V[1].y - V[0].y]],
                         dtype=float)
            element_stiff = (D.T @ D) / (4 * elem.area)
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_stiff[i, j] * v[Vids[i]]
        return w

    def apply_HB_mass(self, v):
        """ Applies the hierarchical mass matrix. """
        w = self.apply_T(v)
        x = self.apply_SS_mass(w)
        return self.apply_T_transpose(x)

    def apply_HB_stiffness(self, v):
        """ Applies the hierarchical stiffness matrix. """
        w = self.apply_T(v)
        x = self.apply_SS_stiffness(w)
        return self.apply_T_transpose(x)

    def apply_boundary_restriction(self, v):
        """ Sets all boundary vertices to zero. """
        w = np.zeros(v.shape)
        for i in range(v.shape[0]):
            w[i] = v[
                i] if not self.triang.vertices[i].on_domain_boundary else 0.0
        return w

    def as_linear_operator(self, method):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(dtype=float,
                              shape=(len(self.triang.vertices),
                                     len(self.triang.vertices)),
                              matvec=lambda x: method(x))

    def as_boundary_restricted_linear_operator(self, method):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(
            dtype=float,
            shape=(len(self.triang.vertices), len(self.triang.vertices)),
            matvec=lambda x: self.apply_boundary_restriction(method(x)))
