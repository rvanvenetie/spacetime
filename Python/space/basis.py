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
        if not deriv:
            result = np.zeros(x.shape[1])
            for elem in self.support:
                bary = elem.to_barycentric_coordinates(x)
                v_index = elem.vertices.index(self.node)
                print(v_index, bary, bary.T[v_index, :])
                result = result + bary[v_index, :] * np.all(bary >= 0, axis=0)
            print("result", result)
            return result
        else:
            result = np.zeros(x.shape)
        for elem in self.support:
            bary = elem.to_barycentric_coordinates(x)
            # Check if this triangle contains the point.
            v_index = elem.vertices.index(self.node)
            if not deriv:
                result = result + bary[v_index, :] * np.all(bary >= 0, axis=0)
            else:
                opp_edge = elem.vertices[(v_index - 1) % 3].as_array() \
                         - elem.vertices[(v_index + 1) % 3].as_array()
                normal = np.array([-opp_edge[1], opp_edge[0]])
                print(normal / (2 * elem.area), np.all(bary > 0, axis=0))
                result -= normal / (2 * elem.area) * np.all(bary > 0, axis=0)

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
