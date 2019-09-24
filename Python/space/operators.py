import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .triangulation_view import TriangulationView


class Operators:
    """ (Temporary) class that defines operators on a triangulation. """
    def __init__(self, triangulation_view):
        assert isinstance(triangulation_view, TriangulationView)
        self.triang = triangulation_view

    def apply_T(self, v):
        """
        Applies the hierarchical-to-single-scale transformation to a vector `v`.

        Arguments:
            v: a `np.array` of length len(self.vertices).

        Returns:
            w: a `np.array` of length len(self.vertices).
        """

        w = np.copy(v)
        for (vi, T) in self.triang.history:
            for gp in T.refinement_edge():
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
        for (vi, T) in reversed(self.triang.history):
            for gp in T.refinement_edge():
                w[gp] = w[gp] + 0.5 * w[vi]
        return w

    def apply_SS_mass(self, v):
        """ Applies the single-scale mass matrix. """
        element_mass = 1.0 / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue

            Vids = elem.vertices_view_idx
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_mass[i, j] * elem.area * v[Vids[i]]
        return w

    def apply_SS_stiffness(self, v):
        """ Applies the single-scale stiffness matrix. """
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue
            Vids = elem.vertices_view_idx
            V = [self.triang.vertices[idx].node for idx in Vids]
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
        for i, vertex in enumerate(self.triang.vertices):
            if not vertex.node.on_domain_boundary:
                w[i] = v[i]
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


def plot_hatfn():
    from .triangulation import InitialTriangulation, to_matplotlib_triangulation
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.bfs()[0].refine()
    T.elem_meta_root.bfs()[4].refine()
    T.elem_meta_root.bfs()[7].refine()
    T.elem_meta_root.bfs()[2].refine()
    T_view = TriangulationView(T)
    matplotlib_triang = to_matplotlib_triangulation(T.elem_meta_root,
                                                    T.vertex_meta_root)

    print(T_view.history)
    I = np.eye(len(T_view.vertices))
    for i in range(len(T_view.vertices)):
        fig = plt.figure()
        fig.suptitle("Hoedfuncties bij vertex %d" % i)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("Nodale basis")
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("Hierarchische basis")
        ax1.plot_trisurf(matplotlib_triang, Z=I[:, i])
        w = op.apply_T(I[:, i])
        ax2.plot_trisurf(matplotlib_triang, Z=w)
        plt.show()


if __name__ == "__main__":
    plot_hatfn()
