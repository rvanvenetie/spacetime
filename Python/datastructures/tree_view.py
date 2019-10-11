from abc import ABC, abstractmethod
from collections import deque

from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree
from .tree import MetaRoot


class NodeViewInterface(MultiNodeViewInterface):
    """ This defines a `view` or `subtree` of an existing underlying tree. """
    dim = 1

    @property
    def node(self):
        return self.nodes[0]


class NodeView(NodeViewInterface, MultiNodeView):
    def __init__(self, nodes, parents=None, children=None):
        if not isinstance(nodes, (list, tuple)): nodes = [nodes]
        super().__init__(nodes=nodes, parents=parents, children=children)


class TreeView(MultiTree):
    def __init__(self, root):
        assert isinstance(root, (NodeViewInterface, MetaRoot))
        if isinstance(root, MetaRoot): root = NodeView([root])
        super().__init__(root=root)

    def uniform_refine(self, max_level=None):
        """ Uniformly refine the multi tree rooted at `self`. """
        if max_level is None:
            call_filter = None
        else:
            call_filter = lambda n: n.level <= max_level
        self.root._deep_refine(call_filter=call_filter)
