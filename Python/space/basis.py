import numpy as np
import quadpy
from functools import lru_cache

from ..datastructures.function import FunctionInterface
from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import NodeView, TreeView


@lru_cache(maxsize=10)
def _get_quadrature_scheme(order):
    # order == 2 * s + 1.
    return quadpy.nsimplex.grundmann_moeller(2, order // 2)


class HierarchicalBasisFunction(FunctionInterface, NodeView):
    order = 1

    @property
    def support(self):
        return self.node.patch

    def eval(self, x, deriv=False):
        """ Evaluate hat function on a number of points `x` at once. """
        assert x.shape[0] == 2
        # If we input a single vector x, we expect a number out.
        if len(x.shape) == 1:
            result = np.zeros(x.shape) if deriv else np.zeros(1)
        else:
            result = np.zeros(x.shape) if deriv else np.zeros(x.shape[1])
        for elem in self.support:
            i = elem.vertices.index(self.node)
            bary = elem.to_barycentric_coordinates(x)
            # mask[j] == True exactly when point x[:,j] is inside elem.
            mask = np.all(bary >= 0, axis=0)
            if not deriv:
                result[mask] = bary[i, mask]
            else:
                V = [elem.vertices[j].as_array() for j in range(3)]
                opp_edge = V[(i - 1) % 3] - V[(i + 1) % 3]
                normal = np.array([-opp_edge[1], opp_edge[0]])
                normal = -normal / (2 * elem.area)
                result[:, mask] = normal.reshape(2, 1)
        # Return singular float if the input was a singular vector x.
        return result if len(x.shape) == 2 else result[0]

    def inner_quad(self, g, g_order=2, deriv=False):
        """ Computes <g, self> or <g, grad self> by quadrature. """
        if not deriv:
            func = lambda x: self.eval(x, deriv) * g(x)
        else:
            func = lambda x: (self.eval(x, deriv) * g(x)).sum(axis=0)
        scheme = _get_quadrature_scheme(g_order + self.order)
        result = 0.0
        for elem in self.support:
            triangle = np.array(
                [elem.vertices[i].as_array() for i in range(3)])
            result += scheme.integrate(func, triangle)
        return result

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates hierarchical basis function tree from the given triang. """
        return TreeView(
            HierarchicalBasisFunction([triangulation.vertex_meta_root]))
