from ..datastructures.function import FunctionInterface
from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import MetaRootView, NodeView


class HierarchicalBasisFunction(FunctionInterface, NodeView):
    def __init__(self, node, parents=None, children=None):
        """ Initializes the Hierarchical basis function.
        
        Args:
          node: the vertex associated to this basis function.
          parents: the parents of this basis function.
          children: the children of this basis function.
        """
        super().__init__(node=node, parents=parents, children=children)

    @property
    def support(self):
        return self.node.patch

    def eval(self, x, deriv=False):
        assert len(x) == 2
        assert not deriv
        for elem in self.support:
            bary = elem.to_barycentric_coordinates(x)
            # Check if this triangle contains the point.
            if bary[0] >= 0 and bary[1] >= 0 and bary[0] + bary[1] <= 1:
                return bary[elem.vertices.index(self.node)]
        return 0

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates a hierarchical basis function tree from the given triang. """
        return MetaRootView(triangulation.vertex_meta_root,
                            HierarchicalBasisFunction)
