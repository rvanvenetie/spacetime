from abc import abstractmethod
from collections import deque

from .tree import NodeInterface


def _replace(i, items, item_i):
    """ Helper function to replace a single item in a list.  """
    result = list(items)
    result[i] = item_i
    return result


class MultiNodeViewInterface(NodeInterface):
    """ Class that represents a multinode interface. """
    __slots__ = []

    # Things to be implemented.
    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def _children(self):
        """ This should return the children as a tuple of length self.dim. """
    @property
    @abstractmethod
    def _parents(self):
        """ This should return the parents as a tuple of length self.dim. """

    # Default implementations.
    @property
    def children(self):
        return self._children

    @property
    def parents(self):
        return self._parents

    @property
    def level(self):
        return sum(n.level for n in self.nodes)

    def is_leaf(self):
        return all(len(self._children[i]) == 0 for i in range(self.dim))

    def is_full(self, i=None):
        if i is None:
            return all(self.is_full(i) for i in range(self.dim))
        else:
            return len(self._children[i]) == len(self.nodes[i].children)

    def is_metaroot(self, i=None):
        """ Returns whether node in any the axes represents a metaroot. """
        if i is None:
            return any(self.is_metaroot(i) for i in range(self.dim))
        else:
            return self.nodes[i].is_metaroot()

    def is_root(self):
        """ Returns whether this multi node view is the root of the mt tree. """
        return all(self.nodes[i].is_metaroot() for i in range(self.dim))

    def coarsen(self):
        return self._coarsen()

    def refine(self,
               i=None,
               children=None,
               call_filter=None,
               make_conforming=False):
        """ Convenience wrapper for refine. """
        if i is None:
            # Concatenate all lists.
            return sum([
                self._refine(i=i,
                             children=children,
                             call_filter=call_filter,
                             make_conforming=make_conforming)
                for i in range(self.dim)
            ], [])
        else:
            assert 0 <= i <= self.dim
            return self._refine(i=i,
                                children=children,
                                call_filter=call_filter,
                                make_conforming=make_conforming)

    def _sparse_refine(self, max_level, weights=None, call_postprocess=None):
        """ Refines this multi tree to a sparse grid multitree.

        Refines such that \sum_i weights[i] * nodes[i].level <= max_level.
        """
        if weights is None: weights = [1] * self.dim
        self._deep_refine(
            call_filter=lambda n: sum(weights[i] * n[i].level
                                      for i in range(self.dim)) <= max_level,
            call_postprocess=call_postprocess)

    def _uniform_refine(self, max_levels=None, call_postprocess=None):
        """ Uniformly refine the multi tree rooted at `self`. """
        call_filter = None
        if isinstance(max_levels, int):
            max_levels = [max_levels] * self.dim
        if max_levels:
            call_filter = lambda n: all(n[i].level <= max_levels[i]
                                        for i in range(self.dim))
        self._deep_refine(call_filter=call_filter,
                          call_postprocess=call_postprocess)

    # Real implementations follow below.
    def _refine(self,
                i,
                children=None,
                call_filter=None,
                make_conforming=False):
        """ Refines the node in the `i`-th coordinate.

        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).

        Args:
          i: The axis we are considering.
          children: If set, the list of children to create. If none, refine
                    all children that exist in the underlying tree.
          call_filter: This function can be used to filter children. It is
            called with the multi-node that is to be created.
        """
        if self.is_full(i): return self._children[i]
        if children is None: children = self.nodes[i].children
        if call_filter is None: call_filter = lambda _: True

        for child_i in children:
            # If this child does not exist in underlying tree, we can stop.
            if child_i not in self.nodes[i].children: continue

            child_nodes = _replace(i, self.nodes, child_i)

            # Skip if this child already exists, or if the filter doesn't pass.
            if child_nodes in (n.nodes for n in self._children[i]) \
                    or not call_filter(child_nodes):
                continue

            # Ensure all the parents of the to be created child exist.
            # These are brothers in various axis of the current node.
            brothers = []
            for j in range(self.dim):
                brothers.append([
                    self._find_brother(
                        _replace(j, child_nodes, child_parent_j), j, i,
                        make_conforming)
                    for child_parent_j in child_nodes[j].parents
                ])

            # Create child.
            child = self.__class__(nodes=child_nodes, parents=brothers)

            # Add child to brothers.
            for j in range(self.dim):
                for brother in brothers[j]:
                    brother._children[j].append(child)

            # Check whether this becomes a child of a metaroot.
            for j in range(self.dim):
                for brother in brothers[j]:
                    if brother.is_metaroot(j):
                        brother.refine(j)

        # Assert metaroot constraint.
        for j in range(self.dim):
            if self.is_metaroot(j) and len(self._children[j]) not in [
                    0, len(self.nodes[j].children)
            ]:
                print('This is a violation of the double tree constraint.')
                print(self, j, self.children[j])
                assert False

        return self._children[i]

    def _coarsen(self):
        """ Removes `self' from the double tree. """
        assert self.is_leaf()
        for i in range(self.dim):
            for parent in self._parents[i]:
                parent._children[i].remove(self)

    def _find_brother(self, nodes, j, i, make_conforming=False):
        """ Finds the given brother in the given axes.

        A brother shares a parent with self in the `i`-axis. That is,
        if self has a parent in the `j`-axis, then the brother would
        have the same parent in the `i`-axis.
        """
        if self.nodes == nodes: return self
        for parent_j in self._parents[j]:
            for sibling_i in parent_j._children[i]:
                if sibling_i.nodes == nodes:
                    return sibling_i

        # We didn't find the brother, lets create it.
        assert make_conforming
        for parent_j in self._parents[j]:
            parent_j.refine(i, children=[nodes[i]], make_conforming=True)

        # Recursive call.
        return self._find_brother(nodes, j, i, make_conforming=False)

    def _union(self, other, call_filter=None, call_postprocess=None):
        """ Deep-copies the node view tree rooted at `other` into self.

        Args:
          other: Root of the other node view tree that we whish to union. We
                 must have self.node == other.node.
          call_filter: This call determines whether a given node of the
              other tree should be inside this subtree.
          call_postprocess: This call will be invoked for every pair
              of nodeview objects. First arg will hold a ref to this tree,
              second arg will hold a ref to the second tree.
        """
        if call_postprocess is None: call_postprocess = lambda _, __: None
        if call_filter is None: call_filter = lambda _: True
        assert isinstance(other, MultiNodeViewInterface)
        assert self.nodes == other.nodes and self.dim == other.dim
        queue = deque([(self, other)])
        my_nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert my_node.nodes == other_node.nodes
            if my_node.marked: continue

            call_postprocess(my_node, other_node)
            my_node.marked = True
            my_nodes.append(my_node)

            for i in range(self.dim):
                filtered_children = list(
                    filter(call_filter, other_node._children[i]))

                # Refine according to the filtered children.
                my_node._refine(
                    i=i, children=[c.nodes[i] for c in filtered_children])

                # Only put children that other_node has as well into the queue.
                for my_child in my_node._children[i]:
                    for other_child in other_node._children[i]:
                        if my_child.nodes == other_child.nodes:
                            queue.append((my_child, other_child))

        # Reset mark field.
        for my_node in my_nodes:
            my_node.marked = False

    def _deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the multitree.

        Args:
          call_filter: This call determines whether a given multinode
            should be inside the subtree.
          call_postprocess: This call will be invoked with a freshly
              created multinode object. Can be used to load data, etc.
        """
        if call_postprocess is None: call_postprocess = lambda _: None
        my_nodes = []
        queue = deque([self])
        while queue:
            my_node = queue.popleft()
            if my_node.marked: continue
            my_nodes.append(my_node)
            call_postprocess(my_node)
            my_node.marked = True
            for i in range(self.dim):
                queue.extend(
                    my_node._refine(i=i,
                                    call_filter=call_filter,
                                    make_conforming=True))

        for my_node in my_nodes:
            my_node.marked = False

    def _bfs(self):
        """ Performs a BFS on the multi tree rooted at `self`.

        This visits the multi nodes in order of their sum of levels.
        """
        queue = deque([self])
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            for i in range(self.dim):
                queue.extend(node._children[i])
        for node in nodes:
            node.marked = False
        return nodes

    def _bfs_kron(self):
        """ Performs a BFS in kron order on the multi tree rooted at `self`.

        This visits the multi nodes in `kron` order. That is, it visits
        the last axis first, then the second-to last axis, etc.

        Suppose we have the full tensor between (0, 1) x (0, 1, 2). Then
        this bfs would return:
            ((0,0), (0,1), (0,2), (1,0), (1,1), (1,2)),
        whereas the normal bfs would return:
            ((0,0), (1,0), (0,1), (1,1), (0,2), (1,2)).
        """

        # Queues holds the children per axis.
        queues = [deque() for _ in range(self.dim)]
        nodes = []
        queues[0].append(self)
        while any(queues):
            # Extract children in reversed order.
            for axis, queue in enumerate(reversed(queues)):
                if queue:
                    node = queue.popleft()
                    break

            if node.marked: continue
            nodes.append(node)
            node.marked = True

            # Only add children in higher axes.
            axis = self.dim - 1 - axis
            for i in range(axis, self.dim):
                queues[i].extend(node._children[i])
        for node in nodes:
            node.marked = False
        return nodes

    def __repr__(self):
        return ' x '.join(map(str, self.nodes))


class MultiNodeView(MultiNodeViewInterface):
    __slots__ = ['nodes', '_parents', '_children', 'marked']

    def __init__(self, nodes, parents=None, children=None):
        """ Creates multi node, with nodes pointing to `single` index node. """
        self.nodes = list(nodes)
        self._parents = parents if parents else [[] for _ in range(self.dim)]
        self._children = children if children else [[]
                                                    for _ in range(self.dim)]
        assert len(self._parents) == self.dim and len(
            self._children) == self.dim and len(nodes) == self.dim

        # Create a marked field useful for bfs/bfs_kron.
        self.marked = False


class MultiTree:
    """ Class that holds the root of the tree. """
    # The fall-back multi node view class.
    mlt_node_cls = MultiNodeView

    def __init__(self, root):
        assert isinstance(root, self.mlt_node_cls)
        assert root.is_root()
        self.root = root

    @classmethod
    def from_metaroots(cls, meta_roots, mlt_node_cls=None):
        """ Initializes the multitree given a tuple of metaroots. """
        assert isinstance(meta_roots, (tuple, list))
        if mlt_node_cls is None: mlt_node_cls = cls.mlt_node_cls
        return cls(mlt_node_cls(meta_roots))

    def bfs(self, include_meta_root=False):
        """ Does a bfs from the root.

        Args:
            include_meta_root: if false, filter out all MetaRoot nodes.
        """
        nodes = self.root._bfs()
        if not include_meta_root:
            nodes = [n for n in nodes if not n.is_metaroot()]
        return nodes

    def bfs_kron(self, include_meta_root=False):
        nodes = self.root._bfs_kron()
        if not include_meta_root:
            nodes = [n for n in nodes if not n.is_metaroot()]
        return nodes

    def deep_copy(self,
                  mlt_tree_cls=None,
                  mlt_node_cls=None,
                  call_postprocess=None):
        """ Copies the current multitree. """
        if mlt_tree_cls is None: mlt_tree_cls = self.__class__
        if mlt_node_cls is None: mlt_node_cls = mlt_tree_cls.mlt_node_cls
        mlt_root = mlt_node_cls(self.root.nodes)
        mlt_root._union(self.root, call_postprocess=call_postprocess)
        mlt_tree = mlt_tree_cls(mlt_root)
        return mlt_tree

    def union(self, other, call_filter=None, call_postprocess=None):
        if isinstance(other, MultiTree): other = other.root
        self.root._union(other,
                         call_filter=call_filter,
                         call_postprocess=call_postprocess)
        return self

    def uniform_refine(self, max_levels=None, call_postprocess=None):
        """ Sparse refines the root of this multi tree view. """
        self.root._uniform_refine(max_levels,
                                  call_postprocess=call_postprocess)

    def sparse_refine(self, max_level, weights=None, call_postprocess=None):
        """ Sparse refines the root of this multi tree view. """
        assert self.root.dim > 1
        self.root._sparse_refine(max_level,
                                 weights=weights,
                                 call_postprocess=call_postprocess)

    def deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep refines the root of this multi tree view. """
        self.root._deep_refine(call_filter, call_postprocess)

    @classmethod
    def make_conforming(cls, nodes):
        """ Creates the smallest conf. mt. tree containing the given nodes. """
        # We will mark all the multinodes that should be in the tree.
        queue = deque(nodes)
        marked_nodes = []
        root = None
        while queue:
            node = queue.popleft()
            if node.marked: continue
            if node.is_root():
                assert root is None
                root = node
            node.marked = True
            marked_nodes.append(node)
            for i in range(node.dim):
                queue.extend(node._parents[i])

        assert root is not None

        # Create a new tree.
        result = cls.from_metaroots(root.nodes)

        # Only copy in nodes that are marked.
        def call_filter(mlt_node):
            return mlt_node.marked

        result.root._union(root, call_filter=call_filter)

        # Unmark the items.
        for node in marked_nodes:
            node.marked = False

        return result
