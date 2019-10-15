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
    @property
    def support(self):
        return self.node.patch

    def eval(self, x, deriv=False):
        assert x.shape[0] == 2
        result = np.zeros(x.shape) if deriv else np.zeros((1, x.shape[1]))
        for elem in self.support:
            i = elem.vertices.index(self.node)
            bary = elem.to_barycentric_coordinates(x)
            mask = np.all(bary >= 0, axis=0)
            if not deriv: result[:, mask] = bary[i, mask]
            else:
                V = [elem.vertices[j].as_array() for j in range(3)]
                opp_edge = V[(i - 1) % 3] - V[(i + 1) % 3]
                normal = np.array([-opp_edge[1], opp_edge[0]])
                result[:, mask] = np.tile(normal[:, np.newaxis], mask.sum())
        return result

    def inner_quad(self, g, deriv=False, order=4):
        """ Computes <g, self> or <g, grad self> by quadrature. """
        def func(x):
            return np.array([
                np.dot(self.eval(x[:, i], deriv), g(x[:, i]))
                for i in range(x.shape[1])
            ])

        result = 0.0
        for elem in self.support:
            triangle = np.array(
                [elem.vertices[i].as_array() for i in range(3)])
            result += _get_quadrature_scheme(order).integrate(func, triangle)
        return result

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates hierarchical basis function tree from the given triang. """
        return TreeView(
            HierarchicalBasisFunction([triangulation.vertex_meta_root]))
