import itertools
from collections import defaultdict, deque

from .tree import MetaRootInterface, NodeInterface
from .tree_view import NodeViewInterface


def _pair(i, item_i, item_not_i):
    """ Helper function to create a pair.
    
    Given coordinate i, returns a pair with item_i in coordinate i,
    and item_not_i in coordinate not i.
    """
    result = [None, None]
    result[i] = item_i
    result[not i] = item_not_i
    return tuple(result)


class DoubleNode:
    """ Class that represents a double node. """
    __slots__ = ['nodes', 'parents', 'children', 'marked']

    def __init__(self, nodes, parents=None, children=None):
        """ Creates double node, with nodes pointing to `single` index node. """
        self.nodes = tuple(nodes)
        self.parents = parents if parents else ([], [])
        self.children = children if children else ([], [])
        assert len(self.parents) == 2 and len(
            self.children) == 2 and len(nodes) == 2

        # Create a marked field useful for bfs/dfs.
        self.marked = False

    def is_leaf(self):
        return len(self.children[0]) == 0 and len(self.children[1]) == 0

    def is_full(self, i=None):
        if i is None:
            return self.is_full(0) and self.is_full(1)
        else:
            return len(self.children[i]) == len(self.nodes[i].children)

    def find_brother(self, i, nodes):
        """ Finds the given brother in the given axis.
        
        A brother shares a parent with self in the `i`-axis.
        """
        if self.nodes == nodes: return self
        for parent_i in self.parents[i]:
            for sibling_i in parent_i.children[i]:
                if sibling_i.nodes == nodes:
                    return sibling_i
        return None

    def find_step_brother(self, i, nodes):
        """ Finds a step brother in the given axis.

        A step brother shares a parent with self in the `other`-axis. That is,
        if self has a parent in the `i`-axis, then the step brother would have
        this parent in the `not i`-axis.
        """
        for parent_i in self.parents[i]:
            for sibling_not_i in parent_i.children[not i]:
                if sibling_not_i.nodes == nodes:
                    return sibling_not_i
        return None

    def refine(self, i, children=None):
        """ Refines the node in the `i`-th coordinate.
        
        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).

        Args:
          i: The axis we are considering.
          children: If set, the list of children to create. If none, refine
                    all children that exist in the underlying tree.

        """
        if self.is_full(i): return self.children[i]
        if children is None: children = self.nodes[i].children

        for child_i in children:
            child_nodes = _pair(i, child_i, self.nodes[not i])

            # Skip if this child has already exists.
            if child_nodes in (n.nodes for n in self.children[i]):
                continue

            # Ensure all the parents of the to be created child exist.
            # These are brothers, in i and not i axis, of the current node.
            brothers = [
                self.find_brother(i,
                                  _pair(i, child_parent_i, child_nodes[not i]))
                for child_parent_i in child_nodes[i].parents
            ]
            step_brothers = [
                self.find_step_brother(
                    not i, _pair(i, child_nodes[i], child_parent_not_i))
                for child_parent_not_i in child_nodes[not i].parents
            ]

            # TODO: Instead of asserting, we could create the (step)brothers.
            assert None not in brothers
            assert None not in step_brothers

            child = self.__class__(child_nodes)
            for brother in brothers:
                child.parents[i].append(brother)
                brother.children[i].append(child)
            for step_brother in step_brothers:
                child.parents[not i].append(step_brother)
                step_brother.children[not i].append(child)

        return self.children[i]

    def coarsen(self):
        """ Removes `self' from the double tree. """
        assert self.is_leaf()
        for i in [0, 1]:
            for parent in self.parents[i]:
                parent.children[i].remove(self)

    def __repr__(self):
        return "{} x {}".format(self.nodes[0], self.nodes[1])


class FrozenDoubleNode(NodeViewInterface):
    """ A double node that is frozen in a single coordinate.
    
    The resulting object acts like a single node in the other coordinate.
    """

    # This should be a lightweight class.
    __slots__ = ['dbl_node', 'i']

    def __init__(self, dbl_node, i):
        """ Freezes the dbl_node in coordinate `not i`. """
        self.dbl_node = dbl_node
        self.i = i

    # Implement the NodeViewInterface method.
    @property
    def node(self):
        return self.dbl_node.nodes[self.i]

    def is_full(self):
        return self.dbl_node.is_full(self.i)

    # Implement the NodeInterface methods.
    def refine(self, children=None):
        dbl_nodes = self.dbl_node.refine(self.i, children=children)
        return [self.__class__(child, self.i) for child in dbl_nodes]

    @property
    def marked(self):
        return self.dbl_node.marked

    @marked.setter
    def marked(self, value):
        self.dbl_node.marked = value

    @property
    def parents(self):
        return [
            self.__class__(parent, self.i)
            for parent in self.dbl_node.parents[self.i]
        ]

    @property
    def children(self):
        return [
            self.__class__(child, self.i)
            for child in self.dbl_node.children[self.i]
        ]

    # Implement some extra methods.
    def union(self, other, call_filter=None, call_postprocess=None):
        """ Deep-copies the singletree rooted at `other` into self. """

        # Only possible if self and other are frozen in the same axis.
        if isinstance(other, FrozenDoubleNode): assert self.i == other.i
        return self._union(other, call_filter, call_postprocess)

    def deep_refine(self, call_filter=None, call_postprocess=None):
        return self._deep_refine(call_filter, call_postprocess)

    def frozen_other_axis(self):
        """ Helper function to get the `self` frozen in the other axis. """
        return self.__class__(self.dbl_node, not self.i)

    def bfs(self, include_meta_root=False):
        nodes = self._bfs()
        if not include_meta_root and isinstance(self.node, MetaRootInterface):
            return nodes[1:]
        else:
            return nodes

    def __repr__(self):
        return '{} x {}'.format(*_pair(self.i, self.node, '_'))

    def __hash__(self):
        return hash((self.dbl_node, self.i))

    def __eq__(self, other):
        if isinstance(other, FrozenDoubleNode):
            return self.i == other.i and self.dbl_node is other.dbl_node
        else:
            return self.node is other


class DoubleTree:
    def __init__(self, root, frozen_dbl_cls=FrozenDoubleNode):
        if isinstance(root, tuple): root = DoubleNode(root)
        assert all(
            isinstance(root.nodes[i], MetaRootInterface) for i in [0, 1])
        assert issubclass(frozen_dbl_cls, FrozenDoubleNode)
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
        if isinstance(mu, FrozenDoubleNode):
            assert not mu.i == i
            return self.fibers[i][mu.node]
        else:
            return self.fibers[i][mu]

    def bfs(self, include_meta_root=False):
        """ Does a bfs from the given double node.
        
        Args:
            i: if set, this assumes we are bfs'ing inside a specific axis.
            include_meta_root: if false, filter out all MetaRoot nodes.
        """
        queue = deque([self.root])
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            # Add the children to the queue.
            queue.extend(node.children[0])
            queue.extend(node.children[1])

        for node in nodes:
            node.marked = False

        if not include_meta_root:
            nodes = [
                n for n in nodes if not any(
                    isinstance(n.nodes[j], MetaRootInterface) for j in [0, 1])
            ]

        return nodes

    def deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the double tree view. 

        Args:
          call_filter: This call determines whether a given double node 
            should be inside the subtree.
          call_postprocess: This call will be invoked with a freshly
              created doublenode object. Can be used to load data, etc.
        """
        if call_filter is None: call_filter = lambda _: True
        if call_postprocess is None: call_postprocess = lambda _: None
        d_nodes = []
        queue = deque([self.root])
        while queue:
            d_node = queue.popleft()
            if d_node.marked: continue
            d_nodes.append(d_node)
            call_postprocess(d_node)
            d_node.marked = True
            for i in [0, 1]:
                for child in d_node.nodes[i].children:
                    if call_filter(_pair(i, child, d_node.nodes[not i])):
                        d_node.refine(i, children=[child])
                queue.extend(d_node.children[i])
        for d_node in d_nodes:
            d_node.marked = False
        self.compute_fibers()

    def deep_copy(self,
                  dbl_node_cls=None,
                  dbl_tree_cls=None,
                  call_postprocess=None):
        """ Copies the current doubletree. """
        if dbl_node_cls is None: dbl_node_cls = self.root.__class__
        if dbl_tree_cls is None: dbl_tree_cls = self.__class__
        double_root = dbl_node_cls(self.root.nodes)
        double_tree = dbl_tree_cls(double_root)
        double_tree.project(0).union(self.project(0),
                                     call_postprocess=call_postprocess)

        for new_node_0, my_node_0 in zip(
                double_tree.project(0).bfs(include_meta_root=True),
                self.project(0).bfs(include_meta_root=True)):

            # We are walking over axis 0, and union'ing in axis 1.
            new_node_1 = new_node_0.frozen_other_axis()
            my_node_1 = my_node_0.frozen_other_axis()
            new_node_1.union(my_node_1, call_postprocess=call_postprocess)

        double_tree.compute_fibers()
        return double_tree

    def uniform_refine(self, max_levels=None):
        """ Uniformly refine this double tree. """
        call_filter = None
        if isinstance(max_levels, int):
            max_levels = (max_levels, max_levels)
        if max_levels:
            call_filter = lambda n: n[0].level <= max_levels[0] and n[
                1].level <= max_levels[1]
        self.deep_refine(call_filter=call_filter)

    def sparse_refine(self, max_level):
        """ Refines this double tree to a sparse grid doubletree. """
        self.deep_refine(
            call_filter=lambda n: n[0].level + n[1].level <= max_level)

    @classmethod
    def full_tensor(cls,
                    meta_root_time,
                    meta_root_space,
                    max_levels=None,
                    dbl_node_cls=DoubleNode,
                    frozen_dbl_cls=FrozenDoubleNode):
        """ Makes a full grid doubletree from the given single trees. """
        double_root = dbl_node_cls((meta_root_time, meta_root_space))
        double_tree = cls(double_root, frozen_dbl_cls=frozen_dbl_cls)
        double_tree.uniform_refine(max_levels)
        return double_tree
