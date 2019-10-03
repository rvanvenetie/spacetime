import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .triangulation_view import TriangulationView


class Operator:
    """ Base class for space operators. """
    def __init__(self, triang=None):
        """ This operator binds to the to a given triangulation(view). """
        self.triang = triang

    def apply(self, v):
        """ Application of the operator the hierarchical basis. """
        assert self.triang
        return self.apply_HB(v)

    def apply_HB(self, v):
        """ Application of the operator the hierarchical basis. 

        Args:
           v: a `np.array` of length len(self.triang.vertices).

        Returns:
            w: a `np.array` of length len(self.triang.vertices).
        """
        """ Applies the hierarchical stiffness matrix. """
        assert len(v) == len(self.triang.vertices)
        w = self.apply_T(v)
        x = self.apply_SS(w)
        return self.apply_T_transpose(x)

    def apply_SS(self, v):
        """Application of the operator to the single-scale basis.

        Abstract method that should be implemented by the derived classess."""
        raise NotImplemented('This function is not implemented')

    def apply_T(self, v):
        """Applies the hierarchical-to-single-scale transformation.  """
        w = np.copy(v)
        for (vi, T) in self.triang.history:
            for gp in T.refinement_edge():
                w[vi] = w[vi] + 0.5 * w[gp]
        return w

    def apply_T_transpose(self, v):
        """Applies the transposed hierarchical-to-single-scale transformation. """
        w = np.copy(v)
        for (vi, T) in reversed(self.triang.history):
            for gp in T.refinement_edge():
                w[gp] = w[gp] + 0.5 * w[vi]
        return w

    def apply_boundary_restriction(self, v):
        """ Sets all boundary vertices to zero. """
        w = np.zeros(v.shape)
        for i, vertex in enumerate(self.triang.vertices):
            if not vertex.node.on_domain_boundary:
                w[i] = v[i]
        return w

    def as_linear_operator(self):
        """ Recasts the application of a this as a scipy LinearOperator. """
        return LinearOperator(dtype=float,
                              shape=(len(self.triang.vertices),
                                     len(self.triang.vertices)),
                              matvec=lambda x: self.apply(x))

    def as_boundary_restricted_linear_operator(self):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(
            dtype=float,
            shape=(len(self.triang.vertices), len(self.triang.vertices)),
            matvec=lambda x: self.apply_boundary_restriction(self.apply(x)))


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
            V = [self.triang.vertices[idx].node for idx in Vids]
            D = np.array([[V[2].x - V[1].x, V[0].x - V[2].x, V[1].x - V[0].x],
                          [V[2].y - V[1].y, V[0].y - V[2].y, V[1].y - V[0].y]],
                         dtype=float)
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
