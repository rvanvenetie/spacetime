
from .multi_tree_view import MultiNodeView, MultiTree
from .tree_view import NodeViewInterface


class DoubleNodeView(MultiNodeView):
    """ Class that represents a double node. """
    __slots__ = []
    dim = 2


class FrozenDoubleNodeView(NodeViewInterface):
    """ A double node that is frozen in a single coordinate.
    
    The resulting object acts like a single node in the other coordinate.
    """
    # This should be a lightweight class.
    __slots__ = ['dbl_node', 'i']

    def __init__(self, dbl_node, i):
        """ Freezes the dbl_node in coordinate `not i`. """
        assert isinstance(dbl_node, DoubleNodeView)
        self.dbl_node = dbl_node
        self.i = i

    # Implement some abstract methods necessary.
    @property
    def nodes(self):
        return [self.dbl_node.nodes[self.i]]

    @property
    def marked(self):
        return self.dbl_node.marked

    @marked.setter
    def marked(self, value):
        self.dbl_node.marked = value

    @property
    def _parents(self):
        return [[
            self.__class__(parent, self.i)
            for parent in self.dbl_node.parents[self.i]
        ]]

    @property
    def _children(self):
        return [[
            self.__class__(child, self.i)
            for child in self.dbl_node.children[self.i]
        ]]

    # Implement some convenience methods.
    @property
    def node(self):
        return self.dbl_node.nodes[self.i]

    def bfs(self, include_meta_root=False):
        nodes = self._bfs()
        if not include_meta_root and self.is_metaroot():
            return nodes[1:]
        else:
            return nodes

    def frozen_other_axis(self):
        """ Helper function to get the `self` frozen in the other axis. """
        return self.__class__(self.dbl_node, not self.i)

    def union(self, other, call_filter=None, call_postprocess=None):
        """ Deep-copies the singletree rooted at `other` into self. """

        # Only possible if self and other are frozen in the same axis.
        if isinstance(other, FrozenDoubleNodeView): assert self.i == other.i
        if isinstance(other, MultiTree): other = other.root
        return self._union(other, call_filter, call_postprocess)

    def _refine(self,
                i,
                children=None,
                call_filter=None,
                make_conforming=False):
        """ Overwrite default call_filter behaviour for frozen tree views. """
        assert i == 0
        if call_filter:
            call_filter_tmp = call_filter
            call_filter = lambda nodes: call_filter_tmp(nodes[self.i])
        self.dbl_node._refine(i=self.i,
                              children=children,
                              call_filter=call_filter,
                              make_conforming=make_conforming)
        return self.children

    def __repr__(self):
        return 'FN({}, {})'.format(self.i, self.node)

    def __hash__(self):
        return hash((self.dbl_node, self.i))

    def __eq__(self, other):
        if isinstance(other, FrozenDoubleNodeView):
            return self.i == other.i and self.dbl_node is other.dbl_node
        else:
            return self.node is other


class DoubleTreeView(MultiTree):
    mlt_node_cls = DoubleNodeView
    frozen_dbl_cls = FrozenDoubleNodeView

    def __init__(self, root):
        assert issubclass(self.frozen_dbl_cls, FrozenDoubleNodeView)
        super().__init__(root=root)
        self.fibers = None

    def compute_fibers(self):
        self.fibers = ({}, {})
        for i in [0, 1]:
            for f_node in self.project(i).bfs(include_meta_root=True):
                self.fibers[not i][f_node.node] = f_node.frozen_other_axis()

    def project(self, i):
        """ Return the list of single nodes in axis i. """
        return self.frozen_dbl_cls(self.root, i)

    def fiber(self, i, mu):
        """ Return the fiber of (single) node mu in axis i.
        
        The fiber is the tree of single-nodes in axis i frozen at coordinate mu
        in the other axis. """
        if not self.fibers:
            self.compute_fibers()

        assert self.fibers[i]
        if isinstance(mu, FrozenDoubleNodeView):
            assert not mu.i == i
            mu = mu.node
        return self.fibers[i][mu]


# Some aliases for legacy reasons
DoubleNode = DoubleNodeView
DoubleTree = DoubleTreeView
FrozenDoubleNode = FrozenDoubleNodeView
