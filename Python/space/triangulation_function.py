import numpy as np

from ..datastructures.multi_tree_function import TreeFunction
from ..space.operators import MassOperator, Operator
from ..space.triangulation import to_matplotlib_triangulation
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
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        matplotlib_triang = to_matplotlib_triangulation(
            triang.elem_tree_view, self)
        fig = fig or plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        ax.plot_trisurf(matplotlib_triang, Z=self_ss)
        if show:
            plt.show()

    @staticmethod
    def interpolate_on(tree, fn):
        result = tree.deep_copy(mlt_tree_cls=TriangulationFunction)
        mass = MassOperator(TriangulationView(tree), dirichlet_boundary=False)
        nodal_eval = np.array([fn(node.node.node.xy) for node in result.bfs()])
        return result.from_array(mass.apply_T_inverse(nodal_eval))

    def L2norm(self):
        triang = TriangulationView(self)
        mass = MassOperator(triang, dirichlet_boundary=False)
        return np.sqrt(self.to_array().T @ mass.apply(self.to_array()))
