import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .triangulation_view import TriangulationView


class Operator:
    """ Base class for space operators. """
    def __init__(self, triang=None, dirichlet_boundary=True):
        """ This operator binds to the to a given triangulation(view). """
        self.triang = triang
        self.dirichlet_boundary = dirichlet_boundary

    def apply(self, v):
        """ Application of the operator the hierarchical basis. """
        assert self.triang
        return self.apply_HB(v.astype(float))

    def apply_HB(self, v):
        """ Application of the operator the hierarchical basis.

        Args:
           v: a `np.array` of length len(self.triang.vertices).

        Returns:
            w: a `np.array` of length len(self.triang.vertices).
        """
        assert len(v) == len(self.triang.vertices)
        if self.dirichlet_boundary:
            v = self.apply_boundary_restriction(v)

        v = self.apply_T(v)
        v = self.apply_SS(v)
        v = self.apply_T_transpose(v)

        if self.dirichlet_boundary:
            v = self.apply_boundary_restriction(v)

        return v

    def apply_SS(self, v):
        """Application of the operator to the single-scale basis.

        Abstract method that should be implemented by the derived classess."""
        raise NotImplementedError('This function is not implemented')

    def apply_T(self, v):
        """Applies the hierarchical-to-single-scale transformation.  """
        w = np.copy(v)
        for (vi, T) in self.triang.history:
            for gp in T.refinement_edge():
                w[vi] = w[vi] + 0.5 * w[gp]
        return w

    def apply_T_transpose(self, v):
        """Applies the transposed hierarchical-to-SS transformation. """
        w = np.copy(v)
        for (vi, T) in reversed(self.triang.history):
            for gp in T.refinement_edge():
                w[gp] = w[gp] + 0.5 * w[vi]
        return w

    def apply_T_inverse(self, v):
        """Applies the single-scale-to-hierarchical transformation. """
        w = np.copy(v)
        for (vi, T) in reversed(self.triang.history):
            for gp in T.refinement_edge():
                w[vi] = w[vi] - 0.5 * w[gp]
        return w

    def apply_boundary_restriction(self, v):
        """ Sets all boundary vertices to zero. """
        w = np.zeros(v.shape)
        for i, vertex in enumerate(self.triang.vertices):
            if not vertex.on_domain_boundary:
                w[i] = v[i]
        return w

    def as_linear_operator(self):
        """ Recasts the application of a this as a scipy LinearOperator. """
        return LinearOperator(dtype=float,
                              shape=(len(self.triang.vertices),
                                     len(self.triang.vertices)),
                              matvec=lambda vec: self.apply(vec))

    def as_boundary_restricted_linear_operator(self):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(dtype=float,
                              shape=(len(self.triang.vertices),
                                     len(self.triang.vertices)),
                              matvec=lambda vec: self.
                              apply_boundary_restriction(self.apply(vec)))


class MassOperator(Operator):
    """ Mass operator.  """
    def apply_SS(self, v):
        """ Applies the single-scale mass matrix.  """
        element_mass = 1.0 / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue

            Vids = elem.vertices_view_idx
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_mass[i, j] * elem.area * v[Vids[i]]
        return w


class StiffnessOperator(Operator):
    """ Stiffness operator. """
    def apply_SS(self, v):
        """ Applies the single-scale stiffness matrix. """
        w = np.zeros(v.shape)
        for elem in self.triang.elements:
            if not elem.is_leaf():
                continue
            Vids = elem.vertices_view_idx
            V = elem.node.vertex_array()
            D = np.array([V[2] - V[1], V[0] - V[2], V[1] - V[0]]).T
            element_stiff = (D.T @ D) / (4 * elem.area)
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_stiff[i, j] * v[Vids[i]]
        return w


def plot_hatfn():
    from .triangulation import InitialTriangulation, to_matplotlib_triangulation
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.bfs()[0].refine()
    T.elem_meta_root.bfs()[4].refine()
    T.elem_meta_root.bfs()[7].refine()
    T.elem_meta_root.bfs()[2].refine()
    T_view = TriangulationView(T.vertex_meta_root)
    op = Operator(T_view)

    matplotlib_triang = to_matplotlib_triangulation(T.elem_meta_root,
                                                    T.vertex_meta_root)
    print(T_view.history)
    I = np.eye(len(T_view.vertices))
    for i in range(len(T_view.vertices)):
        fig = plt.figure()
        fig.suptitle("Hoedfuncties bij vertex %d" % i)
        ax1 = fig.add_subplot(1, 2, 1, projection=Axes3D.name)
        ax1.set_title("Nodale basis")
        ax2 = fig.add_subplot(1, 2, 2, projection=Axes3D.name)
        ax2.set_title("Hierarchische basis")
        ax1.plot_trisurf(matplotlib_triang, Z=I[:, i])
        w = op.apply_T(I[:, i])
        ax2.plot_trisurf(matplotlib_triang, Z=w)
        plt.show()


if __name__ == "__main__":
    plot_hatfn()
