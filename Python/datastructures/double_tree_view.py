import itertools
from collections import defaultdict, deque

from .multi_tree_view import MultiNodeView, MultiNodeViewInterface, MultiTree
from .tree import MetaRoot
from .tree_view import NodeViewInterface


class DoubleNodeView(MultiNodeView):
    """ Class that represents a double node. """
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

    # Overwrite default refine behaviour.
    def refine(self, children=None, call_filter=None):
        dbl_nodes = self._refine(i=0,
                                 children=children,
                                 call_filter=call_filter)
        return [self.__class__(child, self.i) for child in dbl_nodes]

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
    def parents(self):
        return [[
            self.__class__(parent, self.i)
            for parent in self.dbl_node.parents[self.i]
        ]]

    @property
    def children(self):
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
        if not include_meta_root and isinstance(self.node, MetaRoot):
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
        if call_filter:
            call_filter_tmp = call_filter
            call_filter = lambda nodes: call_filter_tmp(nodes[0])
        return self._union(other, call_filter, call_postprocess)

    def _refine(self,
                i,
                children=None,
                call_filter=None,
                call_postprocess=None,
                make_conforming=False):
        assert i == 0
        return self.dbl_node._refine(i=self.i,
                                     children=children,
                                     call_filter=call_filter,
                                     make_conforming=make_conforming)

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
    def __init__(self, root, frozen_dbl_cls=FrozenDoubleNodeView):
        if isinstance(root, (tuple, list)): root = DoubleNode(root)
        assert all(isinstance(root.nodes[i], MetaRoot) for i in [0, 1])
        assert issubclass(frozen_dbl_cls, FrozenDoubleNodeView)
        super().__init__(root=root)
        self.root = root
        self.frozen_dbl_cls = frozen_dbl_cls
        self.compute_fibers()

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
        if isinstance(mu, FrozenDoubleNodeView):
            assert not mu.i == i
            return self.fibers[i][mu.node]
        else:
            return self.fibers[i][mu]

    def uniform_refine(self, max_levels=None):
        self.root._uniform_refine(max_levels)
        self.compute_fibers()

    def sparse_refine(self, max_level):
        self.root._sparse_refine(max_level)
        self.compute_fibers()

    def deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the double tree view. 

        Args:
          call_filter: This call determines whether a given double node 
            should be inside the subtree.
          call_postprocess: This call will be invoked with a freshly
              created doublenode object. Can be used to load data, etc.
        """
        self.root._deep_refine(call_filter, call_postprocess)
        self.compute_fibers()

    @classmethod
    def from_metaroots(cls,
                       meta_root_time,
                       meta_root_space,
                       dbl_node_cls=DoubleNodeView,
                       frozen_dbl_cls=FrozenDoubleNodeView):
        """ Makes a full grid doubletree from the given single trees. """
        assert isinstance(meta_root_time, MetaRoot) and isinstance(
            meta_root_space, MetaRoot)
        double_root = dbl_node_cls((meta_root_time, meta_root_space))
        double_tree = cls(double_root, frozen_dbl_cls=frozen_dbl_cls)
        return double_tree


# Some aliases for legacy reasons
DoubleNode = DoubleNodeView
DoubleTree = DoubleTreeView
FrozenDoubleNode = FrozenDoubleNodeView
