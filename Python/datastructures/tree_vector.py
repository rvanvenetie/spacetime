import numpy as np

from .multi_tree_vector import (MultiNodeVector, MultiNodeVectorInterface,
                                MultiTreeVector)
from .tree import MetaRootInterface
from .tree_view import NodeView, TreeView


class NodeVector(MultiNodeVector, NodeView):
    pass


class TreeVector(MultiTreeVector, TreeView):
    def __init__(self, root):
        if isinstance(root, MetaRootInterface): root = NodeVector([root])
        super().__init__(root)
