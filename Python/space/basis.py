from ..datastructures.function import FunctionInterface
from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import MetaRootView, NodeView


class HierarchicalBasisFunction(FunctionInterface, NodeView):
    __slots__ = ['labda']

    def __init__(self, vertex, parents=None, children=None):
        """ Initializes the Hierarchical basis function.
        
        Args:
          vertex: the vertex associated to this basis function.
          parents: the parents of this basis function.
          children: the children of this basis function.
        """
        super().__init__(vertex, parents, children)
        self.labda = (vertex.level, vertex)

    @property
    def support(self):
        return self.vertex.patch

    @staticmethod
    def from_triangulation(triangulation):
        """ Creates a hierarchical basis function tree from the given triang. """
        return MetaRootView(triangulation.vertex_meta_root,
                            HierarchicalBasisFunction)
