
from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree


class NodeViewInterface(MultiNodeViewInterface):
    """ This defines a `view` or `subtree` of an existing underlying tree. """
    __slots__ = []
    dim = 1

    @property
    def node(self):
        return self.nodes[0]

    @property
    def parents(self):
        """ Simply return the parents in the only dimension we have. """
        return self._parents[0]

    @property
    def children(self):
        """ Simply return the children in the only dimension we have. """
        return self._children[0]

    def _refine(self,
                i,
                children=None,
                call_filter=None,
                make_conforming=False):
        """ Overwrite default call_filter behaviour for single tree views. """
        assert i == 0
        if call_filter:
            call_filter_tmp = call_filter
            call_filter = lambda nodes: call_filter_tmp(nodes[0])
        return super()._refine(i=0,
                               children=children,
                               call_filter=call_filter,
                               make_conforming=make_conforming)


class NodeView(NodeViewInterface, MultiNodeView):
    __slots__ = []

    def __init__(self, nodes, parents=None, children=None):
        if not isinstance(nodes, (list, tuple)): nodes = [nodes]
        super().__init__(nodes=nodes, parents=parents, children=children)


class TreeView(MultiTree):
    mlt_node_cls = NodeView

    @property
    def node(self):
        """ Convenience method for retrieving the node of the root. """
        return self.root.node

    @classmethod
    def from_metaroot(cls, meta_root, node_view_cls=None):
        """ Makes a full grid doubletree from the given single trees. """
        assert meta_root.is_metaroot()
        if node_view_cls is None: node_view_cls = cls.mlt_node_cls
        return cls(node_view_cls(meta_root))

    def uniform_refine(self, max_level=None):
        """ Uniformly refine the multi tree rooted at `self`. """
        if max_level is None:
            call_filter = None
        else:
            call_filter = lambda n: n.level <= max_level
        self.root._deep_refine(call_filter=call_filter)
