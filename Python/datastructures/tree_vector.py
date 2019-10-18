from .multi_tree_vector import MultiNodeVector, MultiTreeVector
from .tree_view import NodeView, TreeView


class NodeVector(MultiNodeVector, NodeView):
    __slots__ = []


class TreeVector(MultiTreeVector, TreeView):
    mlt_node_cls = NodeVector
