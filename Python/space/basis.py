import numpy as np
import quadpy
from functools import lru_cache

from ..datastructures.function import FunctionInterface
from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import NodeView, TreeView


@lru_cache(maxsize=10)
def _get_quadrature_scheme(order):
    return quadpy.line_segment.gauss_patterson(order)


class HierarchicalBasisFunction(FunctionInterface, NodeView):
    order = 1

    @property
    def support(self):
        return self.node.patch

    def eval(self, x, deriv=False):
        assert len(x) == 2
        for elem in self.support:
            bary = elem.to_barycentric_coordinates(x)
            # Check if this triangle contains the point.
            if all(bary >= 0):
                v_index = elem.vertices.index(self.node)
                if not deriv: return bary[v_index]
                else:
                    edge_opposite = elem.vertices[(v_index - 1) % 3].as_array(
                    ) - elem.vertices[(v_index + 1) % 3].as_array()
                    normal = np.array([-edge_opposite[1], edge_opposite[0]])
                    return -normal / (2 * elem.area)

        if not deriv: return 0
        else: return np.zeros(2)

    def inner_quad(self, g, g_order=2, deriv=False):
        """ Computes <g, self> or <g, grad self> by quadrature. """
        def func(x):
            return np.array([
                np.dot(self.eval(x[:, i], deriv), g(x[:, i]))
                for i in range(x.shape[1])
            ])

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
