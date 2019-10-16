import numpy as np

from .multi_tree_vector import (MultiNodeVector, MultiNodeVectorInterface,
                                MultiTreeVector)
from .tree import MetaRoot
from .tree_view import NodeView, NodeViewInterface, TreeView


class NodeVector(MultiNodeVector, NodeView):
    pass


class TreeVector(MultiTreeVector, TreeView):
    def __init__(self, root):
        if not isinstance(root, NodeVector):
            if isinstance(root, MetaRoot):
                root = NodeVector([root])
            elif isinstance(root, NodeViewInterface):
                root = NodeVector([root.node])
            elif isinstance(root, TreeView):
                root = NodeVector([root.root.node])

        assert root.is_metaroot()
        super().__init__(root)
