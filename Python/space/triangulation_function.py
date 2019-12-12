import numpy as np

from ..datastructures.multi_tree_function import TreeFunction
from ..space.operators import MassOperator, Operator
from ..space.triangulation import to_matplotlib_triangulation
from ..space.triangulation_view import TriangulationView


class TriangulationFunction(TreeFunction):
    """ A continuous piecewise affine function defined on a triangulation.

    This is more of a convenience class, with methods that are specific to
    TreeFunctions where the underlying tree is a vertex tree.
    """
    def norm_L2(self):
        """ Calculates the L2 norm of this function. """
        triang = TriangulationView(self)
        mass = MassOperator(triang, dirichlet_boundary=False)
        return np.sqrt(self.to_array().T @ mass.apply(self.to_array()))

    def error_L2(self, g, g_norm_l2, g_quad_order):
        """ Calculates the error in L2 with the given function.

        Args:
          g: lambda of the exact function.
          g_norm_L2: the L2 norm of g.
          g_quad_order: the polynomial order of g, neccessary for quad.
        """

        # Calculate <self, g> using quadrature.
        def call_quad_g(new_node, old_node):
            if new_node.is_metaroot(): return
            new_node.value = old_node.value * old_node.node.inner_quad(
                g=g, g_order=g_quad_order)

        quad_tree = self.deep_copy(call_postprocess=call_quad_g)
        quad_tree_sum = quad_tree.sum()

        # <g - self, g - self> = <g, g> + <self, self> - 2<g, self>.
        result = g_norm_l2**2 + self.norm_L2()**2 - 2 * quad_tree_sum
        return np.sqrt(result)

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
