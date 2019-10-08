from abc import ABC, abstractmethod
from collections import deque

from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree
from .tree import MetaRootInterface


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
        assert isinstance(root, (NodeViewInterface, MetaRootInterface))
        if isinstance(root, MetaRootInterface): root = NodeView([root])
        super().__init__(root=root)

    def deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the tree view. """
        if call_filter:
            call_filter_tmp = call_filter
            call_filter = lambda nodes: call_filter_tmp(nodes[0])
        self.root._deep_refine(call_filter, call_postprocess)

    def uniform_refine(self, max_level):
        self.root._uniform_refine(max_level)
