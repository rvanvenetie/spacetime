import itertools
from collections import defaultdict, deque

from .tree import MetaRoot, NodeAbstract, NodeInterface


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

    def refine(self, i):
        """ Refines the node in the `i`-th coordinate.
        
        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).
        """
        if self.is_full(i): return self.children[i]

        for child_i in self.nodes[i].children:
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

    def union(self, other, i):
        """ Deep-copies the singletree rooted at `other` in axis i into self.

        It is necessary that the singletree `other` is a "full" tree in that
        every node either has *no* children or *all of its possible* children.

        TODO: should get rid of this full-tree condition; it is not necessary.
        """
        queue = deque([(self, other)])
        nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert my_node.nodes[i] == other_node.node
            assert other_node.node.is_leaf() or other_node.node.is_full()
            if my_node.marked: continue
            my_node.marked = True
            nodes.append(my_node)
            if other_node.children:
                my_children = my_node.refine(i)
                assert len(my_children) == len(other_node.children)
                queue.extend(zip(my_children, other_node.children))
        for node in nodes:
            node.marked = False

    def bfs(self, i=None, include_meta_root=False):
        """ Does a bfs from the given double node.
        
        Args:
            i: if set, this assumes we are bfs'ing inside a specific axis.
            include_meta_root: if false, filter out all MetaRoot nodes.
        """
        queue = deque([self])
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            # Add the children to the queue.
            if i is None:
                queue.extend(node.children[0])
                queue.extend(node.children[1])
            else:
                queue.extend(node.children[i])

        for node in nodes:
            node.marked = False

        if not include_meta_root:
            axis = [0, 1] if i is None else [i]
            nodes = [
                node for node in nodes
                if not any(isinstance(node.nodes[j], MetaRoot) for j in axis)
            ]

        return nodes

    def __repr__(self):
        return "{} x {}".format(self.nodes[0], self.nodes[1])


class FrozenDoubleNode(NodeInterface):
    """ A double node that is frozen in a single coordinate.
    
    The resulting object acts like a single node in the other coordinate.
    """

    # This should be a lightweight class.
    __slots__ = ['dbl_node', 'i']

    def __init__(self, dbl_node, i):
        """ Freezes the dbl_node in coordinate `not i`. """
        self.dbl_node = dbl_node
        self.i = i

    def refine(self):
        raise TypeError('FrozenDoubleNode does not support refinement.')

    @property
    def level(self):
        return self.dbl_node[self.i].level

    @property
    def marked(self):
        return self.dbl_node.marked

    @marked.setter
    def marked(self, value):
        self.dbl_node.marked = value

    @property
    def parents(self):
        return [
            FrozenDoubleNode(parent, self.i)
            for parent in self.dbl_node.parents[self.i]
        ]

    @property
    def children(self):
        return [
            FrozenDoubleNode(child, self.i)
            for child in self.dbl_node.children[self.i]
        ]

    @property
    def node(self):
        return self.dbl_node.nodes[self.i]

    def union(self, other):
        return self.dbl_node.union(other, not self.i)

    def coarsen(self):
        return self.dbl_node.coarsen()

    def is_full(self):
        return self.dbl_node.is_full(self.i)

    def bfs(self, include_meta_root=False):
        nodes = self.dbl_node.bfs(self.i, include_meta_root)
        # This returns the items as double nodes, so convert them.
        return [FrozenDoubleNode(node, self.i) for node in nodes]

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
    def __init__(self, root):
        assert all(isinstance(root.nodes[i], MetaRoot) for i in [0, 1])
        self.root = root
        self.compute_fibers()

    def compute_fibers(self):
        self.fibers = ({}, {})
        for i in [0, 1]:
            for node in self.root.bfs(i, include_meta_root=True):
                self.fibers[not i][node.nodes[i]] = FrozenDoubleNode(
                    node, not i)

    def project(self, i):
        """ Return the list of single nodes in axis i. """
        return FrozenDoubleNode(self.root, i)

    def fiber(self, i, mu):
        """ Return the fiber of double-node mu in axis i.
        
        The fiber is the tree of single-nodes in axis i frozen at coordinate mu
        in the other axis. """
        return self.fibers[i][mu]

    def bfs(self, i=None, include_meta_root=False):
        return self.root.bfs(i=i, include_meta_root=include_meta_root)
