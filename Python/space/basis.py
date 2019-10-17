from functools import lru_cache

import numpy as np
import quadpy

from ..datastructures.function import FunctionInterface
from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import NodeView, TreeView
from .triangulation import Vertex


@lru_cache(maxsize=10)
def _get_quadrature_scheme(order):
    # order == 2 * s + 1.
    return quadpy.nsimplex.grundmann_moeller(2, order // 2)


class HierarchicalBasisFunction(FunctionInterface, NodeView):
    order = 1

    def __init__(self, nodes, parents=None, children=None):
        super().__init__(nodes=nodes, parents=parents, children=children)
        assert isinstance(self.node, (Vertex, MetaRoot))

    @property
    def on_domain_boundary(self):
        return self.node.on_domain_boundary

    @property
    def support(self):
        return self.node.patch

    @property
    def patch(self):
        return self.node.patch

    def eval(self, xy, deriv=False):
        """ Evaluate hat function on a number of points `xy` at once. """
        assert xy.shape[0] == 2
        # If we input a single vector xy, we expect a number out.
        if len(xy.shape) == 1:
            result = np.zeros(xy.shape) if deriv else np.zeros(1)
        else:
            result = np.zeros(xy.shape) if deriv else np.zeros(xy.shape[1])
        for elem in self.support:
            i = elem.vertices.index(self.node)
            bary = elem.to_barycentric_coordinates(xy)
            # mask[j] == True exactly when point xy[:,j] is inside elem.
            mask = np.all(bary >= 0, axis=0)
            if not any(mask):
                continue
            if not deriv:
                result[mask] = bary[i, mask]
            else:
                V = elem.vertex_array().T
                opp_edge = V[:, (i - 1) % 3] - V[:, (i + 1) % 3]
                normal = np.array([-opp_edge[1], opp_edge[0]])
                normal = -normal / (2 * elem.area)
                result[:, mask] = normal.reshape(2, 1)
        # Return singular float if the input was a singular vector xy.
        return result if len(xy.shape) == 2 else result[0]

    def inner_quad(self, g, g_order=2, deriv=False):
        """ Computes <g, self> or <g, grad self> by quadrature. """
        if not deriv:
            func = lambda xy: self.eval(xy, deriv) * g(xy)
        else:  # g is a vector field, so take the inner product.
            func = lambda xy: (self.eval(xy, deriv) * g(xy)).sum(axis=0)
        scheme = _get_quadrature_scheme(g_order + self.order)
        result = 0.0
        for elem in self.support:
            triangle = elem.vertex_array()
            result += scheme.integrate(func, triangle)
        return result

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates hierarchical basis function tree from the given triang. """
        return TreeView(
            HierarchicalBasisFunction(triangulation.vertex_meta_root))
