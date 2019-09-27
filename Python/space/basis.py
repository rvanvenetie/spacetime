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

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates a hierarchical basis function tree from the given triang. """
        function_roots = [
            HierarchicalBasisFunction(vertex)
            for vertex in triangulation.vertex_meta_root.roots
        ]
        return MetaRootView(function_roots)
