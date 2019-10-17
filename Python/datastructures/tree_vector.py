import numpy as np

from .double_tree_view import FrozenDoubleNode
from .multi_tree_vector import (MultiNodeVector, MultiNodeVectorInterface,
                                MultiTreeVector)
from .tree import MetaRoot
from .tree_view import NodeView, NodeViewInterface, TreeView


class NodeVector(MultiNodeVector, NodeView):
    __slots__ = []


class TreeVector(MultiTreeVector, TreeView):
    mlt_node_cls = NodeVector
