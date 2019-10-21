import numpy as np

from ..datastructures.multi_tree_function import TreeFunction
from ..space.applicator import Applicator
from ..space.operators import MassOperator, Operator
from ..space.triangulation import (InitialTriangulation,
                                   to_matplotlib_triangulation)
from ..space.triangulation_view import TriangulationView


class TriangulationFunction(TreeFunction):
    """ A piecewise constant function defined on a triangulation. """
    def plot(self, fig=None, show=True, dirichlet_boundary=True):
        # Calculate the triangulation that is associated to the result.
        triang = TriangulationView(self)

        # Convert the result to single scale.
        space_operator = Operator(triang, dirichlet_boundary)
        self_ss = space_operator.apply_T(self.to_array())

        # Plot the result
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        matplotlib_triang = to_matplotlib_triangulation(
            triang.elem_tree_view, self)
        fig = fig or plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(matplotlib_triang, Z=self_ss)
        if show:
            plt.show()

    def from_singlescale_array(self, array):
        triang = TriangulationView(self)
        array_ms = Operator(triang, False).apply_T_inverse(array)
        return self.from_array(array_ms)

    def L2norm(self):
        triang = TriangulationView(self)
        mass = MassOperator(triang, dirichlet_boundary=True)
        return np.sqrt(self.to_array().T @ mass.apply(self.to_array()))
